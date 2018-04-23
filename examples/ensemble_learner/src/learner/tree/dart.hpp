#pragma once

#include "examples/ensemble_learner/src/learner/learner.hpp"

#include "ensemble.hpp"

#include <string>
#include <vector>

namespace flexps {

class DART: public Learner {
  public:
    DART();
    void learn();
    std::map<std::string, float> evaluate();
  protected:
    std::vector<float> get_gradient_vect(std::vector<float>& class_vect, std::vector<float>& estimator_vect, std::string loss_function, int order);
    void update_estimator_vect(float& num_of_drop_trees, std::vector<float>& estimator_vect, std::vector<std::vector<float>>& feat_vect_list);
    float predict(std::vector<float>& data_row);
  private:
    Ensemble ensemble;
    float init_estimator;
    float drop_rate;
};

}