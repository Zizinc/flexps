#include "parameter_server.hpp"

#include "glog/logging.h"

namespace flexps {

ParameterServer::ParameterServer() {
  
}

void ParameterServer::create_kv_tables_for_gbdt(
  Engine& engine,
  int staleness,
  uint64_t nodes_size,
  double num_of_feat, 
  double rank_fraction) {
  
  double size_on_kv_table = (1 / (0.1 * rank_fraction)) * num_of_feat     // quantile sketch
                          + ((1 / rank_fraction) - 1) * 4 * num_of_feat   // grad and hess
                          + 2                                                   // gard sum and count
                          ;

  const uint64_t kMaxKey = (uint64_t) (size_on_kv_table + 0.5);   // add 0.5 to do proper rounding
  LOG(INFO) << "Sized required on KV table: " << kMaxKey;
  
  kv_table_name_to_tid = {
    {"quantile_sketch", 0},
    {"grad_and_hess", 1},
    {"grad_sum_and_count", 2}
  };

  std::vector<third_party::Range> range;
  for (std::map<std::string, int>::iterator it = kv_table_name_to_tid.begin(); it != kv_table_name_to_tid.end(); it++) {
    int tid = it->second;

    range.clear();
    for (int i = 0; i < nodes_size - 1; ++ i) {
      range.push_back({kMaxKey / nodes_size * i, kMaxKey / nodes_size * (i + 1)});
    }
    range.push_back({kMaxKey / nodes_size * (nodes_size - 1), kMaxKey});

    engine.CreateTable<float>(tid, range, ModelType::SSP, StorageType::Map, staleness);
  }

  engine.Barrier();
}

void ParameterServer::get_kv_tables(const Info& info, std::map<std::string, std::unique_ptr<KVClientTable<float>>>* name_to_kv_tables) {
  for (std::map<std::string, int>::iterator it = kv_table_name_to_tid.begin(); it != kv_table_name_to_tid.end(); it++) {
    std::string table_name = it->first;
    int tid = it->second;

    auto table = info.CreateKVClientTable<float>(tid);
    (*name_to_kv_tables)[table_name] = std::move(table);
  }
}

std::vector<uint32_t> ParameterServer::get_tid_vect() {
  std::vector<uint32_t> tid_vect;

  int num_of_table = this->kv_table_name_to_tid.size();
  int tid_array[num_of_table];
  
  int array_idx = 0;
  for (std::map<std::string, int>::iterator it = kv_table_name_to_tid.begin(); it != kv_table_name_to_tid.end(); it++) {
    uint32_t tid = (uint32_t)it->second;
    tid_vect.push_back(tid);
  }
  
  return tid_vect;
}

}