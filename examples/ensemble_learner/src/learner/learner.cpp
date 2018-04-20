#include "learner.hpp"

#include "glog/logging.h"

namespace flexps {

Learner::Learner() {
  
}

void Learner::init(
  std::map<std::string, std::unique_ptr<KVClientTable<float>>>* kv_tables
) {
  this->kv_tables = kv_tables;
}

void Learner::set_training_dataset(DataLoader& data_loader) {
  this->train_data_loader = data_loader;
}

void Learner::set_test_dataset(DataLoader& data_loader) {
  this->test_data_loader = data_loader;
}

void Learner::set_params(std::map<std::string, float>& params) {
  this->params = params;
}

void Learner::set_options(std::map<std::string, std::string>& options) {
  this->options = options;
}

}