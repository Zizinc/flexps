#include "glog/logging.h"
#include "gtest/gtest.h"

#include "examples/ensemble_learner/src/learner/learner_factory.hpp"

namespace flexps {
namespace {

class TestLearnerFactory : public testing::Test {
 public:
  TestLearnerFactory() {}
  ~TestLearnerFactory() {}

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestLearnerFactory, FactoryCreateTest) {
  LearnerFactory learner_factory;
  learner_factory.create_learner(GBDT_LEARNER);
}

}

}