#include "lambda_mart.hpp"
#include "examples/ensemble_learner/src/utilities/math_tools.hpp"
#include "examples/ensemble_learner/src/utilities/util.hpp"

#include "glog/logging.h"

namespace flexps {

LambdaMART::LambdaMART() {

}

void LambdaMART::learn() {
  // Unpack data from DataLoader
  std::vector<float> class_vect = this->train_data_loader.get_class_vect();
  std::vector<int> qid_vect = this->train_data_loader.get_qid_vect();
  std::vector<std::vector<float>> feat_vect_list = this->train_data_loader.get_feat_vect_list();
  std::vector<std::map<std::string, float>> min_max_feat_list = this->train_data_loader.get_min_max_feat_list();
  
  // Initialize
  this->init_estimator = 0.0;
  std::vector<float> estimator_vect(class_vect.size(), this->init_estimator);
  std::vector<float> grad_vect;
  std::vector<float> hess_vect(class_vect.size(), 1.0);
  std::map<int, std::vector<int>> qid_map;
  for (int i = 0; i < qid_vect.size(); i++) {
    auto qid = qid_vect[i];
    qid_map[qid].push_back(i);
  }
  
  float num_of_trees = this->params["num_of_trees"];
  LOG(INFO) << "Begin training LambdaMART, number of trees: " << num_of_trees;
  for (int i = 0; i < num_of_trees; i++) {
    // Compute lambda
    auto lambda_result_map = compute_lambda(qid_map, class_vect, estimator_vect);
    auto lambda_vect = lambda_result_map["lambda_vect"];
    auto weight_vect = lambda_result_map["weight_vect"];

    grad_vect = lambda_vect;
    std::map<std::string, std::vector<float>> additional_vect_map;
    additional_vect_map["weight_vect"] = weight_vect;

    RegressionTree regression_tree;
    regression_tree.init(
      LAMBDAMART_LEARNER,
      feat_vect_list, 
      min_max_feat_list, 
      grad_vect, 
      hess_vect,
      params
    );
    regression_tree.set_kv_tables(kv_tables); //kv_tables is defined in Learner
    regression_tree.set_additional_vect_map(additional_vect_map);
    
    regression_tree.train();
    update_estimator_vect(regression_tree, estimator_vect, feat_vect_list);

    // Step 3: Save the tree to ensemble
    this->ensemble.add_tree(regression_tree);

    // Show train result
    float score = calculate_model_score(qid_map, estimator_vect, class_vect);
    LOG(INFO) << "Node Id = [" << this->params["node_id"] << "], Worker Id = [" << this->params["worker_id"]
      << "]: Train set - Score = [" << score << "], NUM = [" << class_vect.size() << "]";
  }
}

std::map<std::string, float> LambdaMART::evaluate() {
  LOG(INFO) << "Evaluating prediction result for test dataset...";
  std::map<std::string, float> predict_result;

  // Unpack data from DataLoader
  std::vector<float> class_vect = this->train_data_loader.get_class_vect();
  std::vector<int> qid_vect = this->train_data_loader.get_qid_vect();
  std::vector<std::vector<float>> feat_vect_list = this->train_data_loader.get_feat_vect_list();

  // Find qid mapping
  std::map<int, std::vector<int>> qid_map;
  for (int i = 0; i < qid_vect.size(); i++) {
    auto qid = qid_vect[i];
    qid_map[qid].push_back(i);
  }

  // Find prediction scores
  std::vector<float> score_vect;
  for (int i = 0; i < class_vect.size(); i++) {
    std::vector<float> data_row = DataLoader::get_feat_vect_by_row(feat_vect_list, i);
    float score = this->predict(data_row);
    score_vect.push_back(score);
  }
  
  // Calculate ERR (larger is better)
  float err = 0.0;
  for (auto& kv_pair: qid_map) {
    int qid = kv_pair.first;
    auto& idx_vect = kv_pair.second;

    // Retrieve records with the same qid
    std::vector<float> tmp_score_vect;
    std::vector<float> tmp_class_vect;
    for (auto& idx: idx_vect) {
      tmp_score_vect.push_back(score_vect[idx]);
      tmp_class_vect.push_back(class_vect[idx]);
    }
    err += this->err_scorer.calculate_score(tmp_score_vect, tmp_class_vect);
  }
  predict_result["err"] = err;
  predict_result["num"] = score_vect.size();
  return predict_result;
}

std::map<std::string, std::vector<float>> LambdaMART::compute_lambda(std::map<int, std::vector<int>> qid_map, std::vector<float> vect, std::vector<float> score_vect) {
  std::map<std::string, std::vector<float>> result_map;
  std::vector<float> lambda_vect(vect.size(), 0.0);
  std::vector<float> weight_vect(vect.size(), 0.0);

  for (auto& kv_pair: qid_map) {
    int qid = kv_pair.first;
    auto& idx_vect = kv_pair.second;

    // Retrieve records with the same qid
    std::vector<float> tmp_vect;
    std::vector<float> tmp_score_vect;
    for (auto& idx: idx_vect) {
      tmp_vect.push_back(vect[idx]);
      tmp_score_vect.push_back(score_vect[idx]);
    }
    std::map<std::string, std::vector<float>> tmp_result_map = this->dcg_scorer.calculate_lambda_and_weight_vect(qid, tmp_vect, tmp_score_vect);
    std::vector<float> tmp_lambda_vect = tmp_result_map["lambda_vect"];
    std::vector<float> tmp_weight_vect = tmp_result_map["weight_vect"];
    for (int i = 0; i < tmp_lambda_vect.size(); i++) {
      lambda_vect[idx_vect[i]] = tmp_lambda_vect[i];
      weight_vect[idx_vect[i]] = tmp_weight_vect[i];
    }
    result_map["lambda_vect"] = lambda_vect;
    result_map["weight_vect"] = weight_vect;
  }
  return result_map;
}

void LambdaMART::update_estimator_vect(RegressionTree& tree, std::vector<float>& estimator_vect, std::vector<std::vector<float>>& feat_vect_list) {
  for (int i = 0; i < estimator_vect.size(); i++) {
    std::vector<float> data_row = DataLoader::get_feat_vect_by_row(feat_vect_list, i);
    float gamma = tree.predict(data_row);
    estimator_vect[i] += gamma;
  }
}

float LambdaMART::calculate_model_score(std::map<int, std::vector<int>> qid_map, std::vector<float> test_vect, std::vector<float> class_vect) {
  float score = 0.0;

  for (auto& kv_pair: qid_map) {
    int qid = kv_pair.first;
    auto& idx_vect = kv_pair.second;

    // Retrieve records with the same qid
    std::vector<float> tmp_test_vect;
    std::vector<float> tmp_class_vect;
    for (auto& idx: idx_vect) {
      tmp_test_vect.push_back(test_vect[idx]);
      tmp_class_vect.push_back(class_vect[idx]);
    }
    score += this->dcg_scorer.calculate_score(qid, tmp_test_vect, tmp_class_vect);
  }
  return score / class_vect.size();
}

float LambdaMART::predict(std::vector<float>& data_row) {
  float estimator = this->init_estimator;

  for (RegressionTree tree: this->ensemble) {
    estimator += tree.predict(data_row);
  }

  return estimator;
}

}