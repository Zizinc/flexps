#include "glog/logging.h"
#include "gtest/gtest.h"

#include "examples/ensemble_learner/src/channel/channel_util.hpp"
#include "examples/ensemble_learner/src/learner/learner_factory.hpp"
#include "examples/ensemble_learner/src/parameter_server/parameter_server.hpp"

namespace flexps {
namespace {

class TestDART: public testing::Test {
 public:
  TestDART() {}
  ~TestDART() {}

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestDART, SimpleRun) {
  // Remark: running sample on single node single thread
  
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
    train_data_loader.get_feat_vect_list().size(), 
    0.1);

  // Start ML
  MLTask task;
  task.SetWorkerAlloc({{0, 1}});
  const std::vector<uint32_t> tid_vect = parameter_server.get_tid_vect();
  task.SetTables(tid_vect);
  task.SetLambda([& parameter_server, &train_data_loader](const Info& info){
    // Get KV tables
    std::map<std::string, std::unique_ptr<KVClientTable<float>>>* name_to_kv_tables;
    name_to_kv_tables = new std::map<std::string, std::unique_ptr<KVClientTable<float>>>();
    parameter_server.get_kv_tables(info, name_to_kv_tables);

    // Create dart from factory
    LearnerFactory* learner_factory;
    Learner* dart;
    learner_factory = new LearnerFactory;
    dart = learner_factory->create_learner(DART_LEARNER);

    // Config dart
    // Set params
    std::map<std::string, float> params = {
      {"num_of_trees", 5},
      {"max_depth", 1},
      {"complexity_of_leaf", 0.05},
      {"rank_fraction", 0.1},
      {"total_data_num", (float) train_data_loader.get_class_vect().size()},
      {"drop_rate", 0.2},
      // worker info
      {"node_id", 0},
      {"worker_id", 0}
    };
    dart->set_params(params);

    // Set options
    std::map<std::string, std::string> options = {
      {"loss_function", "square_error"}
    };
    dart->set_options(options);

    // Set KV tables
    dart->init(name_to_kv_tables);

    // Set DataLoader
    dart->set_training_dataset(train_data_loader);
    dart->set_test_dataset(train_data_loader);

    // Start learn
    dart->learn();

    // Evaluate
    std::map<std::string, float> predict_result = dart->evaluate();
    LOG(INFO) << "sse = " << predict_result["sse"];
    LOG(INFO) << "num = " << predict_result["num"];
  });
  engine.Run(task);

  engine.StopEverything();
}

}
}