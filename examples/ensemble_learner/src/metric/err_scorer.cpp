#include "err_scorer.hpp"

#include "glog/logging.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <math.h>
#include <string>
#include <vector>

#include "examples/ensemble_learner/src/utilities/util.hpp"

namespace flexps {

ERRScorer::ERRScorer() {
  this->g_max = 16;
}

float ERRScorer::calculate_score(std::vector<float> test_vect) {
  float s = 0.0;
  float p = 1.0;

  for (int i = 0; i < test_vect.size(); i++) {
    float r = calculate_R(test_vect[i]);
    s += p * r / (i + 1);
    p *= (1.0 - r);
  }
  return s;
}

float ERRScorer::calculate_score(std::vector<float> score_vect, std::vector<float> class_vect) {
  std::vector<int> index_vect(score_vect.size(), 0.0);
  
  // Find the sorted sequence (desc)
  for (int i = 0; i < index_vect.size(); i++) {
    index_vect[i] = i;
  }
  sort(index_vect.begin(), index_vect.end(),
    [&](const int& a, const int& b) {
        return (score_vect[a] > score_vect[b]);
    }
  );

  // Order class_vect in that sorted sequence
  std::vector<float> reordered_class_vect(class_vect.size(), 0.0);
  EXPECT_EQ(score_vect.size(), class_vect.size());
  for (int i = 0; i < index_vect.size(); i++) {
    reordered_class_vect[i] = class_vect[index_vect[i]];
  }
  return this->calculate_score(reordered_class_vect);  
}

float ERRScorer::calculate_R(float num) {
  return ((pow(2.0, num) - 1) / this->g_max);
}

}