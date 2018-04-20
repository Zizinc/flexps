#pragma once

#include <map>
#include <string>

#include "examples/ensemble_learner/src/data_loader/data_loader.hpp"

#include "worker/kv_client_table.hpp"

namespace flexps {

class Learner {
  public:
    Learner();
	virtual void learn() = 0;
	virtual std::map<std::string, float> evaluate() = 0;
	void init(
      std::map<std::string, std::unique_ptr<KVClientTable<float>>>* kv_tables
    );
	void set_training_dataset(DataLoader& data_loader);
	void set_test_dataset(DataLoader& data_loader);
	void set_params(std::map<std::string, float>& params);
	void set_options(std::map<std::string, std::string>& options);
  protected:
    DataLoader train_data_loader;
	DataLoader test_data_loader;
	std::map<std::string, std::unique_ptr<KVClientTable<float>>>* kv_tables;
	std::map<std::string, float> params;
	std::map<std::string, std::string> options;
};

}