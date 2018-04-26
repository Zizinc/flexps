#include "gflags/gflags.h"
#include "glog/logging.h"

#include "boost/utility/string_ref.hpp"
#include "base/node_util.hpp"
#include "base/serialization.hpp"
#include "comm/channel.hpp"
#include "comm/mailbox.hpp"
#include "driver/engine.hpp"
#include "driver/simple_id_mapper.hpp"

//#include "io/hdfs_manager.hpp"
#include "lib/batch_data_sampler.cpp"
#include "lib/libsvm_parser.cpp"
#include "worker/kv_client_table.hpp"

// Ensemble Learner src
#include "examples/ensemble_learner/src/channel/channel_util.hpp"
#include "examples/ensemble_learner/src/learner/learner_factory.hpp"
#include "examples/ensemble_learner/src/parameter_server/parameter_server.hpp"

#include <map>
#include <string>
#include <cstdlib>

DEFINE_int32(my_id, -1, "The process id of this program");
DEFINE_string(config_file, "", "The config file path");
DEFINE_int32(num_workers_per_node, 1, "The number of workers per node");
DEFINE_string(run_mode, "", "The running mode of application");
DEFINE_int32(kStaleness, 0, "stalness");

// hdfs mode config
DEFINE_string(hdfs_namenode, "", "The hdfs namenode hostname");
DEFINE_int32(hdfs_namenode_port, -1, "The hdfs namenode port");
DEFINE_string(cluster_train_input, "", "The hdfs train input url");
DEFINE_string(cluster_test_input, "", "The hdfs test input url");

// Local mode config
DEFINE_string(local_train_input, "", "The local train input path");
DEFINE_string(local_test_input, "", "The local test input path");

// gbdt model config
DEFINE_int32(num_of_trees, 1, "The number of trees in forest");
DEFINE_int32(max_depth, 0, "The max depth of each tree");
DEFINE_double(complexity_of_leaf, 0.0, "The complexity of leaf");
DEFINE_double(rank_fraction, 0.1, "The rank fraction of quantile sketch");

namespace flexps {

void Run() {
  CHECK_NE(FLAGS_my_id, -1);
  CHECK(!FLAGS_config_file.empty());
  VLOG(1) << FLAGS_my_id << " " << FLAGS_config_file;

  // 0. Parse config_file
  std::vector<Node> nodes = ParseFile(FLAGS_config_file);
  CHECK(CheckValidNodeIds(nodes));
  CHECK(CheckUniquePort(nodes));
  CHECK(CheckConsecutiveIds(nodes));
  Node my_node = GetNodeById(nodes, FLAGS_my_id);
  LOG(INFO) << my_node.DebugString();

  // Load data (HDFS)
  LOG(INFO) << "Load training dataset";
  /*
  std::vector<std::string> train_data = load_data_from_hdfs(my_node, nodes, FLAGS_cluster_train_input);
  
  std::vector<float> all_class_vect;
  std::vector<std::vector<float>> all_feat_vect_list;
  DataLoader train_data_loader;
  train_data_loader.read_hdfs_to_class_feat_vect(train_data);

  int global_data_num = 0;
  channel_for_balancing_hdfs_data(train_data_loader, global_data_num, my_node, nodes);
  all_feat_vect_list = train_data_loader.get_feat_vect_list();
  
  // Find min and max for each feature
  std::vector<std::map<std::string, float>> min_max_feat_list;
  for (int f_id = 0; f_id < all_feat_vect_list.size(); f_id++) {
    min_max_feat_list.push_back(find_min_max(all_feat_vect_list[f_id]));
  }
  
  std::vector<std::map<std::string, float>> global_min_max_feat_list = channel_for_global_min_max_feat(min_max_feat_list, my_node, nodes);

  // Load testing set
  LOG(INFO) << "Load test dataset";
  std::vector<std::string> test_data = load_data_from_hdfs(my_node, nodes, FLAGS_cluster_test_input);
  DataLoader test_data_loader;
  test_data_loader.read_hdfs_to_class_feat_vect(test_data);
  */

  // Load data (Local)
  //DataLoader train_data_loader("/home/ubuntu/Dataset/40_train.dat");
  DataLoader train_data_loader(FLAGS_local_train_input);
  DataLoader test_data_loader(FLAGS_local_test_input);
  
  int global_data_num = train_data_loader.get_class_vect().size();

  // Find min and max for each feature
  // Store to data loader
  std::vector<std::map<std::string, float>> min_max_feat_list;
  std::vector<std::vector<float>> all_feat_vect_list = train_data_loader.get_feat_vect_list();
  LOG(INFO) << "all_feat_vect_list.size = " << all_feat_vect_list.size();
  for (int f_id = 0; f_id < all_feat_vect_list.size(); f_id++) {
    min_max_feat_list.push_back(find_min_max(all_feat_vect_list[f_id]));
  }
  train_data_loader.set_min_max_feat_list(min_max_feat_list);

  // 1. Start engine
  const int nodeId = my_node.id;
  Engine engine(my_node, nodes);
  engine.StartEverything();

  // 2. Create tables
  // Start ParameterServer
  ParameterServer parameter_server;
  parameter_server.create_kv_tables_for_gbdt(
    engine,
    FLAGS_kStaleness,
    (uint64_t) nodes.size(),
    train_data_loader.get_feat_vect_list().size(), 
    FLAGS_rank_fraction);
  
  // 3. Construct tasks
  LOG(INFO) << "3. construct task";
  int worker_num = (int) FLAGS_num_workers_per_node;
  MLTask task;
  std::vector<WorkerAlloc> worker_alloc;
  for (auto& node : nodes) {
    worker_alloc.push_back({node.id, worker_num});  // each node has worker_num workers
  }
  task.SetWorkerAlloc(worker_alloc);
  const std::vector<uint32_t> tid_vect = parameter_server.get_tid_vect();
  task.SetTables(tid_vect);
  
  std::map<std::string, float> all_predict_result;
  all_predict_result["sse"] = 0;
  all_predict_result["num"] = 0;
  task.SetLambda([& parameter_server, &train_data_loader, &test_data_loader
  	, & global_data_num, & nodeId, & worker_num] (const Info& info) {
  	// Resize dataset by worker id
    auto local_train_data_loader = train_data_loader.create_dataloader_by_worker_id(info.local_id, worker_num);
    auto local_test_data_loader = test_data_loader.create_dataloader_by_worker_id(info.local_id, worker_num);

    // Get KV tables
    std::map<std::string, std::unique_ptr<KVClientTable<float>>>* name_to_kv_tables;
    name_to_kv_tables = new std::map<std::string, std::unique_ptr<KVClientTable<float>>>();
    parameter_server.get_kv_tables(info, name_to_kv_tables);

    // Create gbdt from factory
    LearnerFactory* learner_factory;
    Learner* gbdt;
    learner_factory = new LearnerFactory;
    gbdt = learner_factory->create_learner(GBDT_LEARNER);
  
    // Config gbdt
    // Set params
    std::map<std::string, float> params = {
      {"num_of_trees", FLAGS_num_of_trees},
      {"max_depth", FLAGS_max_depth},
      {"complexity_of_leaf", FLAGS_complexity_of_leaf},
      {"rank_fraction", FLAGS_rank_fraction},
      {"total_data_num", (float) global_data_num},
      {"learning_rate", 1.0},
      // worker info
      {"node_id", nodeId},
      {"worker_id", info.worker_id}
    };
    gbdt->set_params(params);
  
    // Set options
    std::map<std::string, std::string> options = {
      {"loss_function", "square_error"}
    };
    gbdt->set_options(options);
  
    // Set KV tables
    gbdt->init(name_to_kv_tables);
  
    // Set DataLoader
    gbdt->set_training_dataset(local_train_data_loader);
    gbdt->set_test_dataset(local_test_data_loader);
  
    // Start learn
    gbdt->learn();
  
    // Evaluate
    std::map<std::string, float> predict_result = gbdt->evaluate();
    LOG(INFO) << "sse = " << predict_result["sse"];
    LOG(INFO) << "num = " << predict_result["num"];
  });

  // 4. Run tasks
  engine.Run(task);

  // 5. Stop engine
  engine.StopEverything();
}

}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  LOG(INFO) << "argv[0]: " << argv[0];
  flexps::Run();
}