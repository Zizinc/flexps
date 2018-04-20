#include "glog/logging.h"
#include "gtest/gtest.h"

#include "examples/ensemble_learner/src/channel/channel_util.hpp"
#include "examples/ensemble_learner/src/parameter_server/parameter_server.hpp"
#include "examples/ensemble_learner/src/learner/tree/regression_tree.hpp"

namespace flexps {
namespace {

class TestRegressionTree: public testing::Test {
 public:
  TestRegressionTree() {}
  ~TestRegressionTree() {}

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestRegressionTree, FindCandidateSplitsTest) {

  // Define nodes
  std::vector<Node> nodes{
  {0, "localhost", 12353}};
  Node my_node = nodes[0];
  
  // Load data
  DataLoader train_data_loader("/home/ubuntu/Dataset/40_train.dat");
  
  // Find min and max for each feature
  // Store to data loader
  std::vector<std::map<std::string, float>> min_max_feat_list;
  std::vector<std::vector<float>> all_feat_vect_list = train_data_loader.get_feat_vect_list();
  LOG(INFO) << "all_feat_vect_list.size = " << all_feat_vect_list.size();
  for (int f_id = 0; f_id < all_feat_vect_list.size(); f_id++) {
    min_max_feat_list.push_back(find_min_max(all_feat_vect_list[f_id]));
  }
  train_data_loader.set_min_max_feat_list(min_max_feat_list);
  
  // Start Engine
  Engine engine(my_node, nodes);
  engine.StartEverything();
  
  // Start ParameterServer
  ParameterServer parameter_server;
  parameter_server.create_kv_tables_for_gbdt(
    engine,
    0,
    (uint64_t) 1,
    7, 
    0.1);

  // Start ML
  MLTask task;
  task.SetWorkerAlloc({{0, 1}});
  const std::vector<uint32_t> tid_vect = parameter_server.get_tid_vect();
  task.SetTables(tid_vect);
  task.SetLambda([& parameter_server, & train_data_loader](const Info& info){
    // Get KV tables
	std::map<std::string, std::unique_ptr<KVClientTable<float>>>* name_to_kv_tables;
	name_to_kv_tables = new std::map<std::string, std::unique_ptr<KVClientTable<float>>>();
	parameter_server.get_kv_tables(info, name_to_kv_tables);
	
	// Config gbdt
	// Set params
    std::map<std::string, float> params = {
      {"num_of_trees", 1},
      {"max_depth", 0},
      {"complexity_of_leaf", 0.1},
      {"rank_fraction", 0.1},
      {"total_data_num", (float) 40},
      // worker info
      {"node_id", 0},
      {"worker_id", 0}
    };
	
	// Set options
	std::map<std::string, std::string> options = {
      {"loss_function", "square_error"}
    };

    std::vector<float> class_vect = train_data_loader.get_class_vect();
    std::vector<std::vector<float>> feat_vect_list = train_data_loader.get_feat_vect_list();
    std::vector<std::map<std::string, float>> min_max_feat_list = train_data_loader.get_min_max_feat_list();
    std::vector<float> grad_vect = class_vect;
    std::vector<float> hess_vect(class_vect.size(), 1);

    RegressionTree regression_tree;
    regression_tree.init(
      GBDT_LEARNER,
      feat_vect_list, 
      min_max_feat_list, 
      grad_vect, 
      hess_vect,
      params
    );
	regression_tree.set_kv_tables(name_to_kv_tables);
	
	std::vector<std::vector<float>> vect_list = regression_tree.find_candidate_splits();
	
	for (auto vect: vect_list) {
		std::stringstream ss;
		ss << "vect: ";
		for (auto ele: vect) {
			ss << ele << ", ";
		}
		LOG(INFO) << ss.str();
	}
  });
  engine.Run(task);

  engine.StopEverything();
}

TEST_F(TestRegressionTree, RegressionTreeFunctionTest) {

  // Define nodes
  std::vector<Node> nodes{
  {0, "localhost", 12353}};
  Node my_node = nodes[0];
  
  // Load data
  DataLoader train_data_loader("/home/ubuntu/Dataset/40_train.dat");
  
  // Find min and max for each feature
  // Store to data loader
  std::vector<std::map<std::string, float>> min_max_feat_list;
  std::vector<std::vector<float>> all_feat_vect_list = train_data_loader.get_feat_vect_list();
  LOG(INFO) << "all_feat_vect_list.size = " << all_feat_vect_list.size();
  for (int f_id = 0; f_id < all_feat_vect_list.size(); f_id++) {
    min_max_feat_list.push_back(find_min_max(all_feat_vect_list[f_id]));
  }
  train_data_loader.set_min_max_feat_list(min_max_feat_list);
  
  // Start Engine
  Engine engine(my_node, nodes);
  engine.StartEverything();
  
  // Start ParameterServer
  ParameterServer parameter_server;
  parameter_server.create_kv_tables_for_gbdt(
    engine,
    0,
    (uint64_t) 1,
    7, 
    0.1);

  // Start ML
  MLTask task;
  task.SetWorkerAlloc({{0, 1}});
  const std::vector<uint32_t> tid_vect = parameter_server.get_tid_vect();
  task.SetTables(tid_vect);
  task.SetLambda([& parameter_server, & train_data_loader](const Info& info){
    // Get KV tables
	std::map<std::string, std::unique_ptr<KVClientTable<float>>>* name_to_kv_tables;
	name_to_kv_tables = new std::map<std::string, std::unique_ptr<KVClientTable<float>>>();
	parameter_server.get_kv_tables(info, name_to_kv_tables);
	
	// Config gbdt
	// Set params
    std::map<std::string, float> params = {
      {"num_of_trees", 5},
      {"max_depth", 3},
      {"complexity_of_leaf", 0.1},
      {"rank_fraction", 0.1},
      {"total_data_num", (float) 40},
      // worker info
      {"node_id", 0},
      {"worker_id", 0}
    };
	
	// Set options
	std::map<std::string, std::string> options = {
      {"loss_function", "square_error"}
    };

    std::vector<float> class_vect = train_data_loader.get_class_vect();
    std::vector<std::vector<float>> feat_vect_list = train_data_loader.get_feat_vect_list();
    std::vector<std::map<std::string, float>> min_max_feat_list = train_data_loader.get_min_max_feat_list();
    std::vector<float> grad_vect = class_vect;
    std::vector<float> hess_vect(class_vect.size(), 1);

    RegressionTree regression_tree;
    regression_tree.init(
      GBDT_LEARNER,
      feat_vect_list, 
      min_max_feat_list, 
      grad_vect, 
      hess_vect,
      params
    );
	regression_tree.set_kv_tables(name_to_kv_tables);
	
	std::vector<std::vector<float>> candidate_split_vect_list = regression_tree.find_candidate_splits();
	
	regression_tree.find_best_candidate_split(candidate_split_vect_list);
	
	regression_tree.find_predict_val();
  });
  engine.Run(task);

  engine.StopEverything();
}

TEST_F(TestRegressionTree, TrainTreeTest) {

  // Define nodes
  std::vector<Node> nodes{
  {0, "localhost", 12353}};
  Node my_node = nodes[0];
  
  // Load data
  DataLoader train_data_loader("/home/ubuntu/Dataset/40_train.dat");
  
  // Find min and max for each feature
  // Store to data loader
  std::vector<std::map<std::string, float>> min_max_feat_list;
  std::vector<std::vector<float>> all_feat_vect_list = train_data_loader.get_feat_vect_list();
  LOG(INFO) << "all_feat_vect_list.size = " << all_feat_vect_list.size();
  for (int f_id = 0; f_id < all_feat_vect_list.size(); f_id++) {
    min_max_feat_list.push_back(find_min_max(all_feat_vect_list[f_id]));
  }
  train_data_loader.set_min_max_feat_list(min_max_feat_list);
  
  // Start Engine
  Engine engine(my_node, nodes);
  engine.StartEverything();
  
  // Start ParameterServer
  ParameterServer parameter_server;
  parameter_server.create_kv_tables_for_gbdt(
    engine,
    0,
    (uint64_t) 1,
    7, 
    0.1);

  // Start ML
  MLTask task;
  task.SetWorkerAlloc({{0, 1}});
  const std::vector<uint32_t> tid_vect = parameter_server.get_tid_vect();
  task.SetTables(tid_vect);
  task.SetLambda([& parameter_server, & train_data_loader](const Info& info){
    // Get KV tables
	std::map<std::string, std::unique_ptr<KVClientTable<float>>>* name_to_kv_tables;
	name_to_kv_tables = new std::map<std::string, std::unique_ptr<KVClientTable<float>>>();
	parameter_server.get_kv_tables(info, name_to_kv_tables);
	
	// Config gbdt
	// Set params
    std::map<std::string, float> params = {
      {"num_of_trees", 1},
      {"max_depth", 1},
      {"complexity_of_leaf", 0.1},
      {"rank_fraction", 0.1},
      {"total_data_num", (float) 40},
      // worker info
      {"node_id", 0},
      {"worker_id", 0}
    };
	
	// Set options
	std::map<std::string, std::string> options = {
      {"loss_function", "square_error"}
    };

    std::vector<float> class_vect = train_data_loader.get_class_vect();
    std::vector<std::vector<float>> feat_vect_list = train_data_loader.get_feat_vect_list();
    std::vector<std::map<std::string, float>> min_max_feat_list = train_data_loader.get_min_max_feat_list();
    std::vector<float> grad_vect = class_vect;
    std::vector<float> hess_vect(class_vect.size(), 1);

    RegressionTree regression_tree;
    regression_tree.init(
      GBDT_LEARNER,
      feat_vect_list, 
      min_max_feat_list, 
      grad_vect, 
      hess_vect,
      params
    );
	regression_tree.set_kv_tables(name_to_kv_tables);
	
	regression_tree.train();
  });
  engine.Run(task);

  engine.StopEverything();
}

}
}