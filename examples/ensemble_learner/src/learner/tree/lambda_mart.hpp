#pragma once

#include "examples/ensemble_learner/src/learner/learner.hpp"
#include "examples/ensemble_learner/src/metric/dcg_scorer.hpp"
#include "examples/ensemble_learner/src/metric/err_scorer.hpp"

#include "ensemble.hpp"

#include <string>
#include <vector>

namespace flexps {

class LambdaMART: public Learner {
  public:
    LambdaMART();
	  void learn();
	  std::map<std::string, float> evaluate();
  protected:
    std::map<std::string, std::vector<float>> compute_lambda(std::map<int, std::vector<int>> qid_map, std::vector<float> vect, std::vector<float> score_vect);
    void update_estimator_vect(RegressionTree& tree, std::vector<float>& estimator_vect, std::vector<std::vector<float>>& feat_vect_list);
    float calculate_model_score(std::map<int, std::vector<int>> qid_map, std::vector<float> test_vect, std::vector<float> class_vect);
    float predict(std::vector<float>& data_row);
  private:
    Ensemble ensemble;
	  float init_estimator;
	  DCGScorer dcg_scorer;
    ERRScorer err_scorer;
};

}