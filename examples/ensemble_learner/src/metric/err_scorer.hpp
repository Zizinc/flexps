#pragma once

#include <map>
#include <string>
#include <vector>

namespace flexps {

class ERRScorer {
  public:
    ERRScorer();
    float calculate_score(std::vector<float> test_vect);
    float calculate_score(std::vector<float> score_vect, std::vector<float> class_vect);
    float calculate_R(float num);
  protected:
  private:
    float g_max;
};

}