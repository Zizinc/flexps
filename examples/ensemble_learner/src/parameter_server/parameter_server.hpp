#pragma once

#include <map>
#include <string>
#include <vector>

#include "driver/engine.hpp"
#include "worker/kv_client_table.hpp"

namespace flexps {

class ParameterServer {
  public:
    ParameterServer();
    void create_kv_tables_for_gbdt(
      Engine& engine,
      int staleness,
      uint64_t nodes_size,
      double num_of_feat, 
      double rank_fraction
    );
    void get_kv_tables(const Info& info, std::map<std::string, std::unique_ptr<KVClientTable<float>>>* name_to_kv_tables);
    std::vector<uint32_t> get_tid_vect();
  private:
    std::map<std::string, int> kv_table_name_to_tid;
};

}