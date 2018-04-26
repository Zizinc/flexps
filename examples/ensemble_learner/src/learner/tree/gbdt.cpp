#include "gbdt.hpp"
#include "examples/ensemble_learner/src/utilities/math_tools.hpp"
#include "examples/ensemble_learner/src/utilities/timer.hpp"

#include "glog/logging.h"
#include "worker/kv_client_table.hpp"

namespace flexps {

GBDT::GBDT() {
  
}

void GBDT::learn() {
  // Unpack data from DataLoader
  std::vector<float> class_vect = this->train_data_loader.get_class_vect();
  std::vector<std::vector<float>> feat_vect_list = this->train_data_loader.get_feat_vect_list();
  std::vector<std::map<std::string, float>> min_max_feat_list = this->train_data_loader.get_min_max_feat_list();
  
  // Initialize
  this->init_estimator = 0.0;
  std::vector<float> estimator_vect(class_vect.size(), this->init_estimator);
  std::vector<float> grad_vect;
  std::vector<float> hess_vect;
  
  float num_of_trees = this->params["num_of_trees"];
  LOG(INFO) << "Begin training GBDT, number of trees: " << num_of_trees;
  Timer timer;
  timer.start_clock("total_time");
  for (int i = 0; i < num_of_trees; i++) {
    
	  // Step 1: Update residual vect
    grad_vect = get_gradient_vect(class_vect, estimator_vect, this->options["loss_function"], 1);
    hess_vect = get_gradient_vect(class_vect, estimator_vect, this->options["loss_function"], 2);

    // Step 2: Create and train a tree
    RegressionTree regression_tree;
    regression_tree.init(
      GBDT_LEARNER,
      feat_vect_list, 
      min_max_feat_list, 
      grad_vect, 
      hess_vect,
      params
    );
    regression_tree.set_kv_tables(kv_tables); //kv_tables is defined in Learner
    regression_tree.set_timer(&timer);
    regression_tree.train();
    
    update_estimator_vect(regression_tree, estimator_vect, feat_vect_list);
    
    // Step 3: Save the tree to ensemble
    this->ensemble.add_tree(regression_tree);

    // Show train result
    float SSE;
    int NUM;
    get_SSE_and_NUM(estimator_vect, class_vect, SSE, NUM);
    LOG(INFO) << "Node Id = [" << this->params["node_id"] << "], Worker Id = [" << this->params["worker_id"]
      << "]: Train set - SSE = [" << SSE << "], NUM = [" << NUM << "]";
  }
  timer.add_time("total_time");
  LOG(INFO) << "Computation time: " << timer.get_time("computation_time");
  LOG(INFO) << "Communication time: " << timer.get_time("communication_time");
  LOG(INFO) << "Total train time: " << timer.get_time("total_time");
}

std::map<std::string, float> GBDT::evaluate() {
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
std::vector<float> GBDT::get_gradient_vect(std::vector<float>& class_vect, std::vector<float>& estimator_vect, std::string loss_function, int order) {
  std::vector<float> residual_vect;
  for (int i = 0; i < class_vect.size(); i++) {
    float residual = flexps::calculate_gradient(class_vect[i], estimator_vect[i], loss_function, order);
    residual_vect.push_back(residual);
  }
  return residual_vect;
}

void GBDT::update_estimator_vect(RegressionTree& tree, std::vector<float>& estimator_vect, std::vector<std::vector<float>>& feat_vect_list) {
  for (int i = 0; i < estimator_vect.size(); i++) {
    std::vector<float> data_row = DataLoader::get_feat_vect_by_row(feat_vect_list, i);
    float gamma = tree.predict(data_row);
    estimator_vect[i] += gamma;
  }
}

float GBDT::predict(std::vector<float>& data_row) {
  float estimator = this->init_estimator;

  for (RegressionTree tree: this->ensemble) {
    estimator += tree.predict(data_row);
  }

  return estimator;
}

}