#include "glog/logging.h"
#include "gtest/gtest.h"

#include "examples/ensemble_learner/src/channel/channel_util.hpp"

namespace flexps {
namespace {

class TestEnsembleChannelUtil: public testing::Test {
 public:
  TestEnsembleChannelUtil() {}
  ~TestEnsembleChannelUtil() {}

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestEnsembleChannelUtil, FindMaxMinTest) {
  
  // Load data
  DataLoader train_data_loader("/home/ubuntu/Dataset/40_train.dat");
  
  // Define nodes
  std::vector<Node> nodes{
  {0, "localhost", 12353}};
  
  Node my_node = nodes[0];
  
  // Define channel_util
  ChannelUtil channel_util(my_node, nodes);
  
  // Find min and max for each feature
  std::vector<std::map<std::string, float>> min_max_feat_list;
  
  std::vector<std::vector<float>> all_feat_vect_list = train_data_loader.get_feat_vect_list();
  LOG(INFO) << "all_feat_vect_list.size = " << all_feat_vect_list.size();
  for (int f_id = 0; f_id < all_feat_vect_list.size(); f_id++) {
    min_max_feat_list.push_back(find_min_max(all_feat_vect_list[f_id]));
    //LOG(INFO) << "min = " << _min_max_feat_list[f_id]["min"] << ", max = " << _min_max_feat_list[f_id]["max"];
  }
  
  std::vector<std::map<std::string, float>> global_min_max_feat_list = channel_util.channel_for_global_min_max_feat(min_max_feat_list);
  for (int f_id = 0; f_id < global_min_max_feat_list.size(); f_id++) {
    LOG(INFO) << "global min = " << global_min_max_feat_list[f_id]["min"] << ", max = " << global_min_max_feat_list[f_id]["max"];
  }
  
}

}

}