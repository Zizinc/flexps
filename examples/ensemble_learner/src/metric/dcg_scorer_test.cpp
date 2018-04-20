#include "glog/logging.h"
#include "gtest/gtest.h"

#include "examples/ensemble_learner/src/metric/dcg_scorer.hpp"

namespace flexps {
namespace {

class TestDCGScorer : public testing::Test {
 public:
  TestDCGScorer() {}
  ~TestDCGScorer() {}

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestDCGScorer, CalculateDCGTest) {
  DCGScorer dcg_scorer;

  std::vector<float> vect = {0, 0, 0, 1, 1, 0, 1, 1, 0, 0};
  std::vector<float> score_vect = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  auto dcg = dcg_scorer.calculate_dcg(vect);
  EXPECT_NEAR(dcg, 1.466, 0.001);

  auto ideal_dcg = dcg_scorer.calculate_ideal_dcg(vect);
  EXPECT_NEAR(ideal_dcg, 2.562, 0.001);

  auto ndcg = dcg_scorer.calculate_ndcg(1, vect);
  EXPECT_NEAR(ndcg, 0.572, 0.001);

  auto delta_ndcg_1_4 = dcg_scorer.calculate_delta_ndcg(1, vect, 1-1, 4-1);
  EXPECT_NEAR(delta_ndcg_1_4, 0.222, 0.001);

  auto delta_ndcg_1_8 = dcg_scorer.calculate_delta_ndcg(1, vect, 1-1, 8-1);
  EXPECT_NEAR(delta_ndcg_1_8, 0.267, 0.001);
  //auto delta_ndcg_1_100 = dcg_scorer.calculate_delta_ndcg(vect, 1, 100);
  
  auto result_map = dcg_scorer.calculate_lambda_and_weight_vect(1, vect, score_vect);
  EXPECT_NEAR(result_map["lambda_vect"][0], -0.495, 0.001);
  
  for (int i = 0; i < result_map["lambda_vect"].size(); i++) {
  	auto lambda = result_map["lambda_vect"][i];
  	LOG(INFO) << "lambda " << i << " = " << lambda;
  }
  
}

}

}