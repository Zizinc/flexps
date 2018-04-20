#pragma once

#include "gflags/gflags.h"
#include "glog/logging.h"

#include "comm/channel.hpp"

#include "base/node_util.hpp"
#include "comm/mailbox.hpp"
#include "driver/simple_id_mapper.hpp"

#include "examples/ensemble_learner/src/data_loader/data_loader.hpp"
#include "examples/ensemble_learner/src/utilities/math_tools.hpp"

namespace flexps {

class ChannelUtil {
  public:
    ChannelUtil(Node& my_node, std::vector<Node>& nodes);
	void channel_for_balancing_hdfs_data(DataLoader& train_data_loader, int& global_data_num);
    std::vector<std::map<std::string, float>> channel_for_global_min_max_feat(std::vector<std::map<std::string, float>>& min_max_feat_list);
    void channel_for_global_predict_result(const std::map<std::string, float>& local_predict_result);

  protected:
    
  private:
    Node my_node;
    std::vector<Node> nodes;
};

}