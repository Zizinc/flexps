#pragma once

#include <map>
#include <string>
#include <vector>

namespace flexps {

class Scorer {
  public:
    Scorer() {}
    virtual float calculate_score(std::vector<float> test_vect, std::vector<float> class_vect) = 0;

    void set_k(int k) { this->k = k; }
    void set_qid_map(std::map<int, std::vector<int>> qid_map) { this->qid_map = qid_map; }
  protected:
    int k = 10;
    std::map<int, std::vector<int>> qid_map;
};

}