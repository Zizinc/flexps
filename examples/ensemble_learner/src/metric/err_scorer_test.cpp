#include "glog/logging.h"
#include "gtest/gtest.h"

#include "examples/ensemble_learner/src/metric/err_scorer.hpp"

namespace flexps {
namespace {

class TestERRScorer : public testing::Test {
 public:
  TestERRScorer() {}
  ~TestERRScorer() {}

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestERRScorer, CalculateERRTest) {
  ERRScorer err_scorer;
  std::vector<float> test_vect = {3, 2, 4};
  float err = err_scorer.calculate_score(test_vect);
  EXPECT_NEAR(err, 0.63, 0.01);
}

}

}