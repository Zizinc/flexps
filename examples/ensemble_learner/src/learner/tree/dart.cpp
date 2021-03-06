#include <algorithm>
#include <cstdlib>

#include "dart.hpp"
#include "examples/ensemble_learner/src/utilities/math_tools.hpp"

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "worker/kv_client_table.hpp"

namespace flexps {

DART::DART() {
  this->drop_rate = 0.1;
}

void DART::learn() {
  // Unpack data from DataLoader
  std::vector<float> class_vect = this->train_data_loader.get_class_vect();
  std::vector<std::vector<float>> feat_vect_list = this->train_data_loader.get_feat_vect_list();
  std::vector<std::map<std::string, float>> min_max_feat_list = this->train_data_loader.get_min_max_feat_list();
  
  // Initialize
  this->init_estimator = 0.0;
  std::vector<float> estimator_vect(class_vect.size(), this->init_estimator);
  std::vector<float> grad_vect;
  std::vector<float> hess_vect;
  if (this->params.count("drop_rate")) {
    this->drop_rate = this->params["drop_rate"];
  }
  
  float num_of_trees = this->params["num_of_trees"];
  float num_of_drop_trees = 0.0;
  LOG(INFO) << "Begin training DART, number of trees: " << num_of_trees;
  for (int i = 0; i < num_of_trees; i++) {
    
	  // Step 1: Update residual vect
    grad_vect = get_gradient_vect(class_vect, estimator_vect, this->options["loss_function"], 1);
    hess_vect = get_gradient_vect(class_vect, estimator_vect, this->options["loss_function"], 2);

    // Step 2: Create and train a tree
    RegressionTree regression_tree;
    regression_tree.init(
      DART_LEARNER,
      feat_vect_list, 
      min_max_feat_list, 
      grad_vect, 
      hess_vect,
      params
    );
    regression_tree.set_kv_tables(kv_tables); //kv_tables is defined in Learner
    regression_tree.train();
    float factor = (float) 1 / (num_of_drop_trees + 1);
    regression_tree.update_leafs(factor);

    // Step 3: Save the tree to ensemble
    this->ensemble.add_tree(regression_tree);

    // update with Dropout technique
    update_estimator_vect(num_of_drop_trees, estimator_vect, feat_vect_list);
    
    // Show train result
    float SSE;
    int NUM;
    get_SSE_and_NUM(estimator_vect, class_vect, SSE, NUM);
    LOG(INFO) << "Node Id = [" << this->params["node_id"] << "], Worker Id = [" << this->params["worker_id"]
      << "]: Train set - SSE = [" << SSE << "], NUM = [" << NUM << "]";
  }
}

std::map<std::string, float> DART::evaluate() {
  LOG(INFO) << "Evaluating prediction result for test dataset...";
  std::map<std::string, float> predict_result;
  std::vector<float> test_class_vect = this->test_data_loader.get_class_vect();
  std::vector<std::vector<float>> test_feat_vect_list = this->test_data_loader.get_feat_vect_list();
  float SSE = 0.0;

  for (int i = 0; i < test_class_vect.size(); i++) {
    std::vector<float> data_row = DataLoader::get_feat_vect_by_row(test_feat_vect_list, i);
    float predict = this->predict(data_row);
    float error = test_class_vect[i] - predict;
    SSE += error * error;
  }
  
  predict_result["sse"] = SSE;
  predict_result["num"] = test_class_vect.size();
  return predict_result;
}

// Helper function
std::vector<float> DART::get_gradient_vect(std::vector<float>& class_vect, std::vector<float>& estimator_vect, std::string loss_function, int order) {
  std::vector<float> residual_vect;
  for (int i = 0; i < class_vect.size(); i++) {
    float residual = flexps::calculate_gradient(class_vect[i], estimator_vect[i], loss_function, order);
    residual_vect.push_back(residual);
  }
  return residual_vect;
}

void DART::update_estimator_vect(float& num_of_drop_trees, std::vector<float>& estimator_vect, std::vector<std::vector<float>>& feat_vect_list) {
  ASSERT_NE(this->ensemble.get_trees().size(), 0);
  std::vector<int> tree_idx_vect(this->ensemble.get_trees().size());
  for (int i = 0; i < tree_idx_vect.size(); i++) {
    tree_idx_vect[i] = i;
  }

  std::vector<int> drop_idx_vect;
  for (auto& idx: tree_idx_vect) {
    bool drop_flag = ((float) rand() / (RAND_MAX)) < this->drop_rate;
    if (drop_flag) {
      drop_idx_vect.push_back(idx);
    }
  }

  // If drop list is empty, uniformly pick a tree to drop
  if (drop_idx_vect.size() == 0) {
    int drop_idx = rand() % tree_idx_vect.size();
    drop_idx_vect.push_back(drop_idx);
  }

  // Clear estimator vect and train with picked tree
  std::fill(estimator_vect.begin(), estimator_vect.end(), this->init_estimator);
  for (auto& idx: tree_idx_vect) {
    if(std::find(drop_idx_vect.begin(), drop_idx_vect.end(), idx) != drop_idx_vect.end()) {
      continue;
    }
    RegressionTree& tree = this->ensemble.get_tree(idx);
    for (int i = 0; i < estimator_vect.size(); i++) {
      std::vector<float> data_row = DataLoader::get_feat_vect_by_row(feat_vect_list, i);
      estimator_vect[i] += tree.predict(data_row);
    }
  }
  
  // Normalize dropped trees
  float factor = (float) drop_idx_vect.size() / (drop_idx_vect.size() + 1);
  for (auto& idx: drop_idx_vect) {
    RegressionTree& tree = this->ensemble.get_tree(idx);
    tree.update_leafs(factor);
  }

  num_of_drop_trees = drop_idx_vect.size();
}

float DART::predict(std::vector<float>& data_row) {
  float estimator = this->init_estimator;

  for (RegressionTree tree: this->ensemble) {
    estimator += tree.predict(data_row);
  }

  return estimator;
}

}