#include "regression_tree.hpp"
#include "examples/ensemble_learner/src/data_loader/data_loader.hpp"
#include "examples/ensemble_learner/src/utilities/util.hpp"

#include "glog/logging.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <numeric>
#include <math.h>
#include <sstream>

namespace flexps {

RegressionTree::RegressionTree() {
  this->left_child = NULL;
  this->right_child = NULL;
  this->feat_id = -1;
  this->prev_feat_id = -1;
  this->split_val = -1;
  this->max_gain = -std::numeric_limits<float>::infinity();
  this->depth = 0;
  this->predict_val = 0.0;
  this->is_leaf = false;
  this->timer = NULL;
}

void RegressionTree::init(
  LearnerType learner_type,
  std::vector<std::vector<float>>& feat_vect_list, 
  std::vector<std::map<std::string, float>>& min_max_feat_list, 
  std::vector<float>& grad_vect, 
  std::vector<float>& hess_vect,
  std::map<std::string, float>& params
  ) {
  //EXPECT_NE(grad_vect.size(), 0);
  //EXPECT_NE(hess_vect.size(), 0);

  EXPECT_EQ(feat_vect_list[0].size(), grad_vect.size());
  EXPECT_EQ(feat_vect_list[0].size(), hess_vect.size());
  this->learner_type = learner_type;
  this->feat_vect_list = feat_vect_list;
  this->min_max_feat_list = min_max_feat_list;
  this->grad_vect = grad_vect;
  this->hess_vect = hess_vect;
  this->params = params;

  if (this->prev_feat_id == -1) { // root
    this->quantile_sketch_key_vect_list.resize(feat_vect_list.size());
    this->quantile_sketch_val_vect_list.resize(feat_vect_list.size());
    this->candidate_split_vect_list.resize(feat_vect_list.size());
  }
  this->grad_hess_key_vect_list.resize(feat_vect_list.size());
  this->grad_hess_val_vect_list.resize(feat_vect_list.size());
}

void RegressionTree::set_kv_tables(std::map<std::string, std::unique_ptr<KVClientTable<float>>>* kv_tables) {
  this->kv_tables = kv_tables;
}

void RegressionTree::train() {
  std::vector<std::vector<float>> candidate_split_vect_list = find_candidate_splits();
  
  find_best_candidate_split(candidate_split_vect_list);

  if (check_to_stop()) {
    find_predict_val();
  }
  
  reset_kv_tables();
  // Terminate point
  if (check_to_stop()) {
    this->is_leaf = true;
    return;
  }
  
  //Recursively build left and right child
  train_child();
}

std::vector<std::vector<float>> RegressionTree::find_candidate_splits() {
  
  auto & table = (*kv_tables)["quantile_sketch"];
  int ps_key_ptr = 0;
  
  // Step 1: Calculate and push quantile sketch
  // Space needed on PS: (1 / (0.1 * rank_fraction)) * num_of_feat
  std::vector<Key> aggr_push_key_vect;
  std::vector<float> aggr_push_val_vect;
  std::vector<Key> push_key_vect;
  std::vector<float> push_val_vect;

  if (this->timer) {
    this->timer->start_clock("computation_time");
  }

  if (this->prev_feat_id == -1) { // Do all feat
    for (int f_id = 0; f_id < feat_vect_list.size(); f_id++) {
      push_key_vect = push_quantile_sketch(ps_key_ptr, feat_vect_list[f_id], min_max_feat_list[f_id], push_val_vect);
      this->quantile_sketch_key_vect_list[f_id] = push_key_vect;
      this->quantile_sketch_val_vect_list[f_id] = push_val_vect;
      aggr_push_key_vect.insert(aggr_push_key_vect.end(), push_key_vect.begin(), push_key_vect.end());
      aggr_push_val_vect.insert(aggr_push_val_vect.end(), push_val_vect.begin(), push_val_vect.end());
    }
  }
  else { // Do one feat only
    int f_id = this->prev_feat_id;
    push_key_vect = push_quantile_sketch(ps_key_ptr, feat_vect_list[f_id], min_max_feat_list[f_id], push_val_vect);
    // Overwrite push_key_vect if it is not the first run
    push_key_vect = this->quantile_sketch_key_vect_list[f_id];
    this->quantile_sketch_val_vect_list[f_id] = push_val_vect;
    aggr_push_key_vect.insert(aggr_push_key_vect.end(), push_key_vect.begin(), push_key_vect.end());
    aggr_push_val_vect.insert(aggr_push_val_vect.end(), push_val_vect.begin(), push_val_vect.end());
  }
  
  if (this->timer) {
    this->timer->add_time("computation_time");
    this->timer->start_clock("communication_time");
  }

  EXPECT_NE(aggr_push_key_vect.size(), 0);
  table->Add(aggr_push_key_vect, aggr_push_val_vect);
  table->Clock();

  // Step 2: Pull global quantile sketch result and find split candidates
  std::vector<Key> pull_key_vect;
  std::vector<float> pull_val_vect;

  if (this->prev_feat_id == -1) {  // Do all feat
    for (int f_id = 0; f_id < this->quantile_sketch_key_vect_list.size(); f_id++) {
      std::vector<Key> key_vect = this->quantile_sketch_key_vect_list[f_id];
      pull_key_vect.insert(pull_key_vect.end(), key_vect.begin(), key_vect.end());
    }
  }
  else { // Do one feat only
    int f_id = this->prev_feat_id;
    std::vector<Key> key_vect = this->quantile_sketch_key_vect_list[f_id];
    pull_key_vect.insert(pull_key_vect.end(), key_vect.begin(), key_vect.end());
  }

  table->Get(pull_key_vect, &pull_val_vect);
  table->Clock();

  if (this->timer) {
    this->timer->add_time("communication_time");
    this->timer->start_clock("computation_time");
  }

  if (this->prev_feat_id == -1) {
    int sketch_num_per_feat = pull_val_vect.size() / feat_vect_list.size(); 
    for (int f_id = 0; f_id < feat_vect_list.size(); f_id++) {
      std::vector<float> sketch_hist_vect(pull_val_vect.begin() + (f_id * sketch_num_per_feat), pull_val_vect.begin() + ((f_id + 1) * sketch_num_per_feat));
      std::vector<float> candidate_split_vect = find_candidate_split(sketch_hist_vect, min_max_feat_list[f_id]);
      this->candidate_split_vect_list[f_id] = candidate_split_vect;
    }
  }
  else {
    int f_id = this->prev_feat_id;
    std::vector<float> sketch_hist_vect = pull_val_vect;
    std::vector<float> candidate_split_vect = find_candidate_split(sketch_hist_vect, min_max_feat_list[f_id]);
    this->candidate_split_vect_list[f_id] = candidate_split_vect;
  }

  if (this->timer) {
    this->timer->add_time("computation_time");
  }

  return this->candidate_split_vect_list;
}

void RegressionTree::find_best_candidate_split(std::vector<std::vector<float>>& candidate_split_vect_list) {
  
  auto & table = (*kv_tables)["grad_and_hess"];
  int ps_key_ptr = 0;
  
  // Step 3: Calculate local grad and hess for each candidate and push the result to ps
  // Space needed on PS: ((1 / rank_fraction) - 1) * 4 * num_of_feat
  std::vector<Key> aggr_push_key_vect;
  std::vector<float> aggr_push_val_vect;
  std::vector<Key> push_key_vect;
  std::vector<float> push_val_vect;

  if (this->timer) {
    this->timer->start_clock("computation_time");
  }

  for (int f_id = 0; f_id < feat_vect_list.size(); f_id++) {
    // Ordering of push_key_vect:
    // left grad, left hess, right grad, right hess
    push_key_vect = push_local_grad_hess(ps_key_ptr, feat_vect_list[f_id], candidate_split_vect_list[f_id], grad_vect, hess_vect, push_val_vect);
    this->grad_hess_key_vect_list[f_id] = push_key_vect;
    this->grad_hess_val_vect_list[f_id] = push_val_vect;
    aggr_push_key_vect.insert(aggr_push_key_vect.end(), push_key_vect.begin(), push_key_vect.end());
    aggr_push_val_vect.insert(aggr_push_val_vect.end(), push_val_vect.begin(), push_val_vect.end());
  }
  EXPECT_NE(aggr_push_key_vect.size(), 0);

  if (this->timer) {
    this->timer->add_time("computation_time");
    this->timer->start_clock("communication_time");
  }

  table->Add(aggr_push_key_vect, aggr_push_val_vect);
  table->Clock();
  
  // Step 4: Pull global grad and hess and find the best split
  std::vector<Key> pull_key_vect;
  std::vector<float> pull_val_vect;

  for (int f_id = 0; f_id < this->grad_hess_key_vect_list.size(); f_id++) {
    std::vector<Key> key_vect = this->grad_hess_key_vect_list[f_id];
    pull_key_vect.insert(pull_key_vect.end(), key_vect.begin(), key_vect.end());
    
  }
  table->Get(pull_key_vect, &pull_val_vect);
  table->Clock();

  if (this->timer) {
    this->timer->add_time("communication_time");
    this->timer->start_clock("computation_time");
  }

  int grad_hess_num_per_feat = ((1 / this->params["rank_fraction"]) - 1) * 4;
  for (int f_id = 0; f_id < feat_vect_list.size(); f_id++) {
    std::vector<float> grad_hess_vect(pull_val_vect.begin() + (f_id * grad_hess_num_per_feat), pull_val_vect.begin() + ((f_id + 1) * grad_hess_num_per_feat));
    std::map<std::string, float> best_split = find_best_split(grad_hess_vect, this->params["complexity_of_leaf"]);
    
    // Update best split info for this node
    if (best_split["best_split_gain"] > this->max_gain) {
      this->max_gain = best_split["best_split_gain"];
      this->feat_id = f_id;
      this->split_val = candidate_split_vect_list[f_id][best_split["best_candidate_id"]];
    }
  }

  if (this->timer) {
    this->timer->add_time("computation_time");
  }
}

void RegressionTree::find_predict_val() {
  auto & table = (*kv_tables)["grad_sum_and_count"];
  int ps_key_ptr = 0;
  
  // Push sum and count
  std::vector<Key> aggr_push_key_vect;
  std::vector<float> aggr_push_val_vect;
  std::vector<Key> push_key_vect;
  std::vector<float> push_val_vect;

  if (this->timer) {
    this->timer->start_clock("computation_time");
  }

  push_key_vect = push_local_grad_sum(ps_key_ptr, push_val_vect);
  
  this->local_grad_sum_key_vect_list = push_key_vect;
  this->local_grad_sum_val_vect_list = push_val_vect;
    
  aggr_push_key_vect.insert(aggr_push_key_vect.end(), push_key_vect.begin(), push_key_vect.end());
  aggr_push_val_vect.insert(aggr_push_val_vect.end(), push_val_vect.begin(), push_val_vect.end());
  
  EXPECT_NE(aggr_push_key_vect.size(), 0);

  if (this->timer) {
    this->timer->add_time("computation_time");
    this->timer->start_clock("communication_time");
  }

  table->Add(aggr_push_key_vect, aggr_push_val_vect);
  table->Clock();
  
  // Pull sum and count
  std::vector<Key> pull_key_vect;
  std::vector<float> pull_val_vect;
  pull_key_vect = this->local_grad_sum_key_vect_list;
  table->Get(pull_key_vect, &pull_val_vect);
  table->Clock();
  
  if (this->timer) {
    this->timer->add_time("communication_time");
  }

  this->predict_val = pull_val_vect[0] / pull_val_vect[1];
}

void RegressionTree::reset_kv_tables() {
  if (this->timer) {
    this->timer->start_clock("computation_time");
  }

  // Reset quantile_sketch table
  std::vector<Key> aggr_push_key_vect;
  std::vector<float> aggr_push_val_vect;
  std::vector<float> push_val_vect;
  if (this->prev_feat_id == -1) {
    for (int i = 0; i < this->quantile_sketch_key_vect_list.size(); i++) {
      std::vector<float> inv_val_vect;
      for (int j = 0; j < this->quantile_sketch_key_vect_list[i].size(); j++) {
        inv_val_vect.push_back(this->quantile_sketch_val_vect_list[i][j] * -1.0);
      }
      push_val_vect = inv_val_vect;
      aggr_push_key_vect.insert(aggr_push_key_vect.end(), quantile_sketch_key_vect_list[i].begin(), quantile_sketch_key_vect_list[i].end());
      aggr_push_val_vect.insert(aggr_push_val_vect.end(), push_val_vect.begin(), push_val_vect.end());
    }
  }
  else {
    int f_id = this->prev_feat_id;
    std::vector<float> inv_val_vect;
    for (int j = 0; j < this->quantile_sketch_key_vect_list[f_id].size(); j++) {
      inv_val_vect.push_back(this->quantile_sketch_val_vect_list[f_id][j] * -1.0);
    }
    push_val_vect = inv_val_vect;
    aggr_push_key_vect.insert(aggr_push_key_vect.end(), quantile_sketch_key_vect_list[f_id].begin(), quantile_sketch_key_vect_list[f_id].end());
    aggr_push_val_vect.insert(aggr_push_val_vect.end(), push_val_vect.begin(), push_val_vect.end());
  }
  
  if (this->timer) {
    this->timer->add_time("computation_time");
    this->timer->start_clock("communication_time");
  }

  (*kv_tables)["quantile_sketch"]->Add(aggr_push_key_vect, aggr_push_val_vect);
  (*kv_tables)["quantile_sketch"]->Clock();
  
  std::vector<float> pull_val_vect;
  (*kv_tables)["quantile_sketch"]->Get(aggr_push_key_vect, &pull_val_vect);
  (*kv_tables)["quantile_sketch"]->Clock();
  EXPECT_NEAR(std::accumulate(pull_val_vect.begin(), pull_val_vect.end(), 0.0), 0.0, 0.00001);
  
  if (this->timer) {
    this->timer->add_time("communication_time");
    this->timer->start_clock("computation_time");
  }

  // Reset grad_and_hess table
  aggr_push_key_vect.clear();
  aggr_push_val_vect.clear();
  
  for (int i = 0; i < this->grad_hess_key_vect_list.size(); i++) {
    std::vector<float> inv_val_vect;
    for (int j = 0; j < this->grad_hess_key_vect_list[i].size(); j++) {
      inv_val_vect.push_back(this->grad_hess_val_vect_list[i][j] * -1.0);
    }
    push_val_vect = inv_val_vect;
    aggr_push_key_vect.insert(aggr_push_key_vect.end(), grad_hess_key_vect_list[i].begin(), grad_hess_key_vect_list[i].end());
    aggr_push_val_vect.insert(aggr_push_val_vect.end(), push_val_vect.begin(), push_val_vect.end());
  }
  
  if (this->timer) {
    this->timer->add_time("computation_time");
    this->timer->start_clock("communication_time");
  }

  (*kv_tables)["grad_and_hess"]->Add(aggr_push_key_vect, aggr_push_val_vect);
  (*kv_tables)["grad_and_hess"]->Clock();
  
  (*kv_tables)["grad_and_hess"]->Get(aggr_push_key_vect, &pull_val_vect);
  (*kv_tables)["grad_and_hess"]->Clock();
  // FIXME: When the grad is too large, cannot exactly reset due to discard of floating points
  //EXPECT_NEAR(std::accumulate(pull_val_vect.begin(), pull_val_vect.end(), 0.0), 0.0, 0.00001);

  if (this->timer) {
    this->timer->add_time("communication_time");
  }

  // Reset grad_sum_and_count table
  if (check_to_stop()) {
    aggr_push_key_vect.clear();
    aggr_push_val_vect.clear();
  
    std::vector<float> inv_val_vect;
    for (int i = 0; i < this->local_grad_sum_key_vect_list.size(); i++) {
      inv_val_vect.push_back(this->local_grad_sum_val_vect_list[i] * -1.0);
    }
    push_val_vect = inv_val_vect;
    aggr_push_key_vect.insert(aggr_push_key_vect.end(), local_grad_sum_key_vect_list.begin(), local_grad_sum_key_vect_list.end());
    aggr_push_val_vect.insert(aggr_push_val_vect.end(), push_val_vect.begin(), push_val_vect.end());
  
    if (this->timer) {
      this->timer->start_clock("communication_time");
    }

    EXPECT_NE(aggr_push_key_vect.size(), 0);
    (*kv_tables)["grad_sum_and_count"]->Add(aggr_push_key_vect, aggr_push_val_vect);
    (*kv_tables)["grad_sum_and_count"]->Clock();
    (*kv_tables)["grad_sum_and_count"]->Get(aggr_push_key_vect, &pull_val_vect);
    (*kv_tables)["grad_sum_and_count"]->Clock();
    // FIXME: Cannot exactly reset due to discard of floating points
    //EXPECT_NEAR(std::accumulate(pull_val_vect.begin(), pull_val_vect.end(), 0.0), 0.0, 0.00001);

    if (this->timer) {
      this->timer->add_time("communication_time");
    }
  }
}

void RegressionTree::train_child() {
  // Initialize dataset
  std::vector<float> left_grad_vect, right_grad_vect;
  std::vector<float> left_hess_vect, right_hess_vect;
  std::vector<std::vector<float>> left_feat_vect_list, right_feat_vect_list;
  EXPECT_NE(this->feat_id, -1);
  EXPECT_NE(this->split_val, -1);
  DataLoader::split_dataset_by_feat_val(
    this->feat_id,
    this->split_val,
    grad_vect,
    left_grad_vect,
    right_grad_vect,
    hess_vect,
    left_hess_vect,
    right_hess_vect,
    feat_vect_list,
    left_feat_vect_list,
    right_feat_vect_list
  );
  
  std::vector<std::map<std::string, float>> left_min_max_feat_list = min_max_feat_list;
  std::vector<std::map<std::string, float>> right_min_max_feat_list = min_max_feat_list;
  left_min_max_feat_list[this->feat_id]["max"] = this->split_val;
  right_min_max_feat_list[this->feat_id]["min"] = this->split_val;
  
  // Config
  this->left_child = new RegressionTree;
  this->right_child = new RegressionTree;
  this->left_child->information_caching(
    this->feat_id,
    this->quantile_sketch_key_vect_list,
    this->quantile_sketch_val_vect_list,
    this->candidate_split_vect_list
    );
  this->right_child->information_caching(
    this->feat_id,
    this->quantile_sketch_key_vect_list,
    this->quantile_sketch_val_vect_list,
    this->candidate_split_vect_list
    );
  this->left_child->init(
    learner_type,
    left_feat_vect_list, 
    left_min_max_feat_list, 
    left_grad_vect, 
    left_hess_vect,
    params
    );
  this->right_child->init(
    learner_type,
    right_feat_vect_list, 
    right_min_max_feat_list, 
    right_grad_vect, 
    right_hess_vect,
    params
  );
  this->left_child->set_timer(this->timer);
  this->right_child->set_timer(this->timer);
  this->left_child->set_kv_tables(this->kv_tables);
  this->right_child->set_kv_tables(this->kv_tables);
  this->left_child->set_depth(this->depth + 1);
  this->right_child->set_depth(this->depth + 1);

  switch(this->learner_type) {
    case LAMBDAMART_LEARNER:
      split_and_set_additional_vect_map();
      break;
  }
  
  this->left_child->train();
  this->right_child->train();
}

float RegressionTree::predict(std::vector<float>& vect) {
  if (this->is_leaf) {
    float learning_rate = 1.0;
    if (this->params.count("learning_rate")) {
      learning_rate = this->params["learning_rate"];
    }
    return learning_rate * this->predict_val;
  }
  else {
    if (vect[this->feat_id] < this->split_val) {
      return this->left_child->predict(vect);
    }
    else {
      return this->right_child->predict(vect);
    }
  }
}

void RegressionTree::update_leafs(float factor) {
  if (this->is_leaf) {
    this-> predict_val *= factor;
  }
  else {
    this->left_child->update_leafs(factor);
    this->right_child->update_leafs(factor);
  }
}

void RegressionTree::information_caching(
        int prev_feat_id,
        std::vector<std::vector<Key>> quantile_sketch_key_vect_list,
        std::vector<std::vector<float>> quantile_sketch_val_vect_list,
        std::vector<std::vector<float>> candidate_split_vect_list
        ) {
  this->prev_feat_id = prev_feat_id;
  this->quantile_sketch_key_vect_list = quantile_sketch_key_vect_list;
  this->quantile_sketch_val_vect_list = quantile_sketch_val_vect_list;
  this->candidate_split_vect_list = candidate_split_vect_list;
}

// Helper function
std::vector<Key> RegressionTree::push_quantile_sketch(int& ps_key_ptr, std::vector<float>& feat_vect, std::map<std::string, float>& min_max_feat, std::vector<float>& _push_val_vect) {
  float rank_fraction = this->params["rank_fraction"];
  float total_data_num = this->params["total_data_num"];
  float min = min_max_feat["min"];
  float max = min_max_feat["max"];
  max += 0.0001; // To include largest element

  // Step 1: Create histogram
  int rank_num = (int) 1.0 / rank_fraction;
  std::vector<float> rank_fraction_vect;
  for (int i = 1; i < rank_num; i++) {
    rank_fraction_vect.push_back(rank_fraction * i);
  }

  float sketch_fraction = 0.1 * rank_fraction;
  int sketch_num = (int) 1.0 / sketch_fraction;
  std::vector<float> sketch_bin_vect;
  for (int i = 0; i <= sketch_num; i++) {
    sketch_bin_vect.push_back(sketch_fraction * i);
  }

  // Step 2: Accumulate histogram
  std::vector<float> sketch_hist_vect(sketch_num, 0.0);
  for(int i = 0; i < feat_vect.size(); i++) {
    float weight_val = (feat_vect[i] - min) / (max - min);
    for(int j = 0; j < sketch_bin_vect.size() - 1; j ++) {
      if (sketch_bin_vect[j] <= weight_val && weight_val < sketch_bin_vect[j + 1]) {
        sketch_hist_vect[j] += 1;
        break;
      }
    }
  }

  std::vector<float> push_val_vect = sketch_hist_vect;
  std::vector<Key> push_key_vect(push_val_vect.size());
  std::iota(push_key_vect.begin(), push_key_vect.end(), ps_key_ptr);
  ps_key_ptr += push_key_vect.size();

  _push_val_vect = push_val_vect;
  return push_key_vect;
}

std::vector<float> RegressionTree::find_candidate_split(std::vector<float>& sketch_hist_vect, std::map<std::string, float>& min_max_feat) {
  float rank_fraction = this->params["rank_fraction"];
  float total_data_num = this->params["total_data_num"];
  float min = min_max_feat["min"];
  float max = min_max_feat["max"];
  max += 0.0001; // To include largest element
  
  int rank_num = (int) 1.0 / rank_fraction;
  std::vector<float> rank_fraction_vect;
  for (int i = 1; i < rank_num; i++) {
    rank_fraction_vect.push_back(rank_fraction * i);
  }
  float rank_bin_width = rank_fraction * total_data_num;

  float sketch_fraction = 0.1 * rank_fraction;
  int sketch_num = (int) 1.0 / sketch_fraction;

  int sketch_candidate_range = 2 * sketch_num / rank_num;

  std::vector<float> candidate_vect;
  float prev_fraction = 0.0;
  for (int c_id = 0; c_id < rank_fraction_vect.size(); c_id++) {
    float argmax_fraction = prev_fraction;
    float best_approx = std::numeric_limits<float>::infinity();

    float cur_candidate_aggr_num = (c_id + 1) * rank_bin_width;
    for (int s_id = 0; s_id < sketch_candidate_range; s_id++) {
      float cur_fraction = prev_fraction + s_id * sketch_fraction;
      float cur_fraction_sum = std::accumulate(sketch_hist_vect.begin(), sketch_hist_vect.begin() + (int)(cur_fraction / sketch_fraction), 0.0);

      if (fabs(cur_fraction_sum - cur_candidate_aggr_num) <= best_approx) {
        best_approx = fabs(cur_fraction_sum - cur_candidate_aggr_num);
        argmax_fraction = cur_fraction;
      }
    }
    prev_fraction = argmax_fraction;
    candidate_vect.push_back(argmax_fraction * (max - min) + min);
  }
  return candidate_vect;
}

std::vector<Key> RegressionTree::push_local_grad_hess(int& ps_key_ptr, std::vector<float>& feat_vect, std::vector<float>& candidate_split_vect
      , std::vector<float>& grad_vect, std::vector<float>& hess_vect, std::vector<float>& _push_val_vect) {
  int candidate_num_per_feat = candidate_split_vect.size();
  std::vector<float> left_grad_val_vect(candidate_num_per_feat, 0.0);
  std::vector<float> left_hess_val_vect(candidate_num_per_feat, 0.0);
  std::vector<float> right_grad_val_vect(candidate_num_per_feat, 0.0);
  std::vector<float> right_hess_val_vect(candidate_num_per_feat, 0.0);

  // Step 1: Accumulate left and right grad/hess val vect
  for (int col_id = 0; col_id < feat_vect.size(); col_id++) {
    float feat_val = feat_vect[col_id];
    for (int c_id = 0; c_id < candidate_num_per_feat; c_id++) {
      float candidate_split_val = candidate_split_vect[c_id];

      if (feat_val < candidate_split_val) {
        left_grad_val_vect[c_id] += grad_vect[col_id];
        left_hess_val_vect[c_id] += hess_vect[col_id];
      }
      else {
        right_grad_val_vect[c_id] += grad_vect[col_id];
        right_hess_val_vect[c_id] += hess_vect[col_id];
      }
    }
  }

  // Step 2: Push grad and hess to PS
  std::vector<float> push_val_vect;
  push_val_vect.insert(push_val_vect.end(), left_grad_val_vect.begin(), left_grad_val_vect.end());
  push_val_vect.insert(push_val_vect.end(), left_hess_val_vect.begin(), left_hess_val_vect.end());
  push_val_vect.insert(push_val_vect.end(), right_grad_val_vect.begin(), right_grad_val_vect.end());
  push_val_vect.insert(push_val_vect.end(), right_hess_val_vect.begin(), right_hess_val_vect.end());

  std::vector<Key> push_key_vect(push_val_vect.size());
  std::iota(push_key_vect.begin(), push_key_vect.end(), ps_key_ptr);
  ps_key_ptr += push_key_vect.size();
  
  _push_val_vect = push_val_vect;
  return push_key_vect;
}

std::map<std::string, float> RegressionTree::find_best_split(std::vector<float>& grad_hess_vect, float complexity_of_leaf) {
  // Step 1: Decode vect from PS
  std::vector<float> left_grad_val_vect(grad_hess_vect.begin(), grad_hess_vect.begin() + (grad_hess_vect.size() / 4));
  std::vector<float> left_hess_val_vect(grad_hess_vect.begin() + (grad_hess_vect.size() / 4), grad_hess_vect.begin() + (grad_hess_vect.size() / 2));
  std::vector<float> right_grad_val_vect(grad_hess_vect.begin() + (grad_hess_vect.size() / 2), grad_hess_vect.begin() + (grad_hess_vect.size() * 3 / 4));
  std::vector<float> right_hess_val_vect(grad_hess_vect.begin() + (grad_hess_vect.size() * 3 / 4), grad_hess_vect.begin() + grad_hess_vect.size());

  // Step 2: Find best split
  int best_candidate_id = -1;
  float best_split_gain = -std::numeric_limits<float>::infinity();
  int candidate_num = grad_hess_vect.size() / 4;
  for (int c_id = 0; c_id < candidate_num; c_id++) {
    float left_grad_sum = left_grad_val_vect[c_id];
    float left_hess_sum = left_hess_val_vect[c_id];
    float right_grad_sum = right_grad_val_vect[c_id];
    float right_hess_sum = right_hess_val_vect[c_id];
    float grad_sum = left_grad_sum + right_grad_sum;
    float hess_sum = left_hess_sum + right_hess_sum;

    float left_result = ((left_grad_sum * left_grad_sum) / (left_hess_sum + complexity_of_leaf));
    float right_result = ((right_grad_sum * right_grad_sum) / (right_hess_sum + complexity_of_leaf));
    float orig_result = ((grad_sum * grad_sum) / (hess_sum + complexity_of_leaf));
    
    float split_gain = left_result + right_result - orig_result;
    if (split_gain > best_split_gain) {
      best_split_gain = split_gain;
      best_candidate_id = c_id;
      }
  }
  std::map<std::string, float> res;
  res["best_candidate_id"] = (float) best_candidate_id;
  res["best_split_gain"] = best_split_gain;

  return res;
}

std::vector<Key> RegressionTree::push_local_grad_sum(int& ps_key_ptr, std::vector<float>& _push_val_vect) {
  float local_grad_sum = std::accumulate(grad_vect.begin(), grad_vect.end(), 0.0);
  float local_grad_num = grad_vect.size();

  switch (this->learner_type) {
    case LAMBDAMART_LEARNER:
      auto& weight_vect = this->additional_vect_map["weight_vect"];
      EXPECT_NE(weight_vect.size(), 0);
      local_grad_num = std::accumulate(weight_vect.begin(), weight_vect.end(), 0.0);
      break;
  }

  std::vector<float> push_val_vect;
  push_val_vect.push_back(local_grad_sum);
  push_val_vect.push_back(local_grad_num);

  std::vector<Key> push_key_vect(push_val_vect.size());
  std::iota(push_key_vect.begin(), push_key_vect.end(), ps_key_ptr);
  ps_key_ptr += push_key_vect.size();

  _push_val_vect = push_val_vect;
  return push_key_vect;
}

bool RegressionTree::check_to_stop() {
  // 1. Check depth
  if (this->depth >= this->params["max_depth"]) {
    return true;
  }

  return false;
}

void RegressionTree::set_depth(int depth) {
  this->depth = depth;
}

void RegressionTree::split_and_set_additional_vect_map() {
  std::map<std::string, std::vector<float>> l_map, r_map;
  for (auto& kv: this->additional_vect_map) {
    auto key = kv.first;
    auto val_vect = kv.second;
    std::vector<float> l_val_vect, r_val_vect;

    auto& feat_vect = this->feat_vect_list[this->feat_id];
    EXPECT_EQ(feat_vect.size(), val_vect.size());
    for (int i = 0; i < feat_vect.size(); i++) {
      if (feat_vect[i] < this->split_val) {
        l_val_vect.push_back(val_vect[i]);
      }
      else {
        r_val_vect.push_back(val_vect[i]);
      }
    }
    l_map[key] = l_val_vect;
    r_map[key] = r_val_vect;
  }
  if (this->left_child == NULL || this->right_child == NULL) {
    LOG(INFO) << "left or right child has not been initialized";
    exit(-1);
  }
  this->left_child->set_additional_vect_map(l_map);
  this->right_child->set_additional_vect_map(r_map);
}

}