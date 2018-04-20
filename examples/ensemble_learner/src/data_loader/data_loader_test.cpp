#include "glog/logging.h"
#include "gtest/gtest.h"

#include "examples/ensemble_learner/src/data_loader/data_loader.hpp"

namespace flexps {
namespace {

class TestEnsembleDataLoader : public testing::Test {
 public:
  TestEnsembleDataLoader() {}
  ~TestEnsembleDataLoader() {}

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestEnsembleDataLoader, SvmRankReadTest) {
  DataLoader data_loader("svm_rank", "/home/ubuntu/Dataset/lambdaMART_sample.dat");
  data_loader.print_loaded_data();
}

}

}