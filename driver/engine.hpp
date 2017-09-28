#pragma once

#include <memory>
#include <vector>
#include <algorithm>

#include "base/node.hpp"
#include "worker/worker_helper_thread.hpp"
#include "worker/app_blocker.hpp"
#include "worker/simple_range_manager.hpp"
#include "worker/model_init_thread.hpp"
#include "server/server_thread.hpp"
#include "server/server_thread_group.hpp"
#include "server/map_storage.hpp"
#include "server/vector_storage.hpp"
#include "server/ssp_model.hpp"
#include "server/bsp_model.hpp"
#include "server/asp_model.hpp"
#include "server/sparse_ssp_model.hpp"
#include "server/abstract_sparse_ssp_recorder.hpp"
#include "server/unordered_map_sparse_ssp_recorder.hpp"
#include "server/vector_sparse_ssp_recorder.hpp"
#include "comm/mailbox.hpp"
#include "comm/sender.hpp"
#include "driver/ml_task.hpp"
#include "driver/simple_id_mapper.hpp"
#include "driver/info.hpp"
#include "driver/worker_spec.hpp"

#include "server/sparse_ssp_model_2.hpp"
#include "server/abstract_sparse_ssp_recorder_2.hpp"
#include "server/unordered_map_sparse_ssp_recorder_2.hpp"

namespace flexps {

enum class ModelType {
  SSP, BSP, ASP, SparseSSP
};
enum class StorageType {
  Map, Vector
};
enum class SparseSSPRecorderType {
  None, Map, Vector
};


class Engine {
 public:
  Engine(const Node& node, const std::vector<Node>& nodes) : node_(node), nodes_(nodes) {}
  /*
   * The flow of starting the engine:
   * 1. Create a mailbox
   * 2. Start Sender
   * 3. Create ServerThreads, WorkerHelperThreads and ModelInitThread
   * 4. Register the threads to mailbox through ThreadsafeQueue
   * 5. Start the mailbox: bind and connect to all other nodes
   */
  void StartEverything();
  void CreateMailbox();
  void StartServerThreads();
  void StartWorkerHelperThreads();
  void StartModelInitThread();
  void StartMailbox();
  void StartSender();

  /*
   * The flow of stopping the engine:
   * 1. Stop the Sender
   * 2. Stop the mailbox: by Barrier() and then exit
   * 3. The mailbox will stop the corresponding registered threads
   * 4. Stop the ServerThreads, WorkerHelperThreads and ModelInitThread
   */
  void StopEverything();
  void StopServerThreads();
  void StopWorkerHelperThreads();
  void StopModelInitThread();
  void StopSender();
  void StopMailbox();
 
  void Barrier();
  WorkerSpec AllocateWorkers(const std::vector<WorkerAlloc>& worker_alloc);
  template<typename Val>
  void CreateTable(uint32_t table_id, const std::vector<third_party::Range>& ranges,
      ModelType model_type, StorageType storage_type, int model_staleness = 0, int speculation = 0,
      SparseSSPRecorderType sparse_ssp_recorder_type = SparseSSPRecorderType::None);
  void InitTable(uint32_t table_id, const std::vector<uint32_t>& worker_ids);
  void Run(const MLTask& task);

 private:
  void RegisterRangeManager(uint32_t table_id, const std::vector<third_party::Range>& ranges);

  std::map<uint32_t, SimpleRangeManager> range_manager_map_;
  // nodes
  Node node_;
  std::vector<Node> nodes_;
  // mailbox
  std::unique_ptr<SimpleIdMapper> id_mapper_;
  std::unique_ptr<Mailbox> mailbox_;
  std::unique_ptr<Sender> sender_;
  // worker elements
  std::unique_ptr<AppBlocker> app_blocker_;
  std::unique_ptr<WorkerHelperThread> worker_helper_thread_;
  std::unique_ptr<ModelInitThread> model_init_thread_;
  // server elements
  std::unique_ptr<ServerThreadGroup> server_thread_group_;
};

template<typename Val>
void Engine::CreateTable(uint32_t table_id,
    const std::vector<third_party::Range>& ranges,
    ModelType model_type, StorageType storage_type, int model_staleness, int speculation,
    SparseSSPRecorderType sparse_ssp_recorder_type) {
  RegisterRangeManager(table_id, ranges);
  CHECK(server_thread_group_);

  CHECK(id_mapper_);
  auto server_thread_ids = id_mapper_->GetAllServerThreads();
  CHECK_EQ(ranges.size(), server_thread_ids.size());

  for (auto& server_thread : *server_thread_group_) {
    std::unique_ptr<AbstractStorage> storage;
    std::unique_ptr<AbstractModel> model;
    // Set up storage
    if (storage_type == StorageType::Map) {
      storage.reset(new MapStorage<Val>());
    } else if (storage_type == StorageType::Vector){
      auto it = std::find(server_thread_ids.begin(), server_thread_ids.end(), server_thread->GetServerId());
      storage.reset(new VectorStorage<Val>(ranges[it - server_thread_ids.begin()]));
    } else {
      CHECK(false) << "Unknown storage_type";
    }
    // Set up model
    if (model_type == ModelType::SSP) {
      model.reset(new SSPModel(table_id, std::move(storage), model_staleness, server_thread_group_->GetReplyQueue()));
    } else if (model_type == ModelType::BSP) {
      model.reset(new BSPModel(table_id, std::move(storage), server_thread_group_->GetReplyQueue()));
    } else if (model_type == ModelType::ASP) {
      model.reset(new ASPModel(table_id, std::move(storage), server_thread_group_->GetReplyQueue()));
    } else if (model_type == ModelType::SparseSSP) {
      std::unique_ptr<AbstractSparseSSPRecorder2> recorder;
      CHECK(sparse_ssp_recorder_type != SparseSSPRecorderType::None);
      if (sparse_ssp_recorder_type == SparseSSPRecorderType::Map) {
        recorder.reset(new UnorderedMapSparseSSPRecorder2(model_staleness, speculation));
      } else if (sparse_ssp_recorder_type == SparseSSPRecorderType::Vector) {
        CHECK(false);
        // auto it = std::find(server_thread_ids.begin(), server_thread_ids.end(), server_thread->GetServerId());
        // recorder.reset(new VectorSparseSSPRecorder2(model_staleness, speculation, ranges[it - server_thread_ids.begin()]));
      } else {
        CHECK(false);
      }
      model.reset(new SparseSSPModel2(table_id, std::move(storage), std::move(recorder), server_thread_group_->GetReplyQueue(), model_staleness, speculation));
    } else {
      CHECK(false) << "Unknown model_type";
    }
    server_thread->RegisterModel(table_id, std::move(model));
  }
}

}  // namespace flexps
