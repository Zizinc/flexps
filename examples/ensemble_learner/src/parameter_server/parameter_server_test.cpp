#include "glog/logging.h"
#include "gtest/gtest.h"

#include "driver/engine.hpp"
#include "examples/ensemble_learner/src/parameter_server/parameter_server.hpp"

namespace flexps {
namespace {

class TestParameterServer: public testing::Test {
 public:
  TestParameterServer() {}
  ~TestParameterServer() {}

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestParameterServer, CreateKVTableForGBDT) {
  Node node{0, "localhost", 12353};
  Engine engine(node, {node});
  engine.StartEverything();
  
  ParameterServer parameter_server;
  parameter_server.create_kv_tables_for_gbdt(
    engine,
    0,
    (uint64_t) 1,
    7, 
    0.1);

  MLTask task;
  task.SetWorkerAlloc({{0, 1}});
  const std::vector<uint32_t> tid_vect = parameter_server.get_tid_vect();
  task.SetTables(tid_vect);
  //task.SetTables({0,1,2});
  task.SetLambda([& parameter_server](const Info& info){
    std::map<std::string, std::unique_ptr<KVClientTable<float>>>* name_to_kv_tables;
	name_to_kv_tables = new std::map<std::string, std::unique_ptr<KVClientTable<float>>>();
		
	parameter_server.get_kv_tables(info, name_to_kv_tables);
	
	for (std::map<std::string, std::unique_ptr<KVClientTable<float>>>::iterator it = name_to_kv_tables->begin(); it != name_to_kv_tables->end(); it++) {
      std::string table_name = it->first;
      LOG(INFO) << " table name = " << table_name;
    }
	
  });
  engine.Run(task);

  
  engine.StopEverything();
}

}

}