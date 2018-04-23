#pragma once

#include "examples/ensemble_learner/src/learner/learner.hpp"

#include "ensemble.hpp"

#include <string>
#include <vector>

namespace flexps {

class GBDT: public Learner {
  public:
    GBDT();
    void learn();
    std::map<std::string, float> evaluate();
  protected:
    std::vector<float> get_gradient_vect(std::vector<float>& class_vect, std::vector<float>& estimator_vect, std::string loss_function, int order);
    void update_estimator_vect(RegressionTree& tree, std::vector<float>& estimator_vect, std::vector<std::vector<float>>& feat_vect_list);
    float predict(std::vector<float>& data_row);
  private:
    Ensemble ensemble;
    float init_estimator;
};

}