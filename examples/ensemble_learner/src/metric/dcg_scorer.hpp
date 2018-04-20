#pragma once

#include <map>
#include <string>
#include <vector>

namespace flexps {

class DCGScorer {
  public:
    DCGScorer();
    float calculate_dcg(std::vector<float> vect);
    float calculate_ideal_dcg(std::vector<float> vect);
    float get_ideal_dcg(int qid, std::vector<float> vect);
    float calculate_ndcg(int qid, std::vector<float> vect);
    float calculate_delta_ndcg(int qid, std::vector<float> vect, int position_i, int position_j);
    float calculate_rho(std::vector<float> vect, std::vector<float> score_vect, int position_i, int position_j);
    float calculate_lambda(int qid, std::vector<float> vect, std::vector<float> score_vect, int position_i, int position_j);
    std::map<std::string, std::vector<float>> calculate_lambda_and_weight_vect(int qid, std::vector<float> vect, std::vector<float> score_vect);
    void update_ideal_dcg_map(int qid, std::vector<float> vect);
    float calculate_score(int qid, std::vector<float> test_vect, std::vector<float> class_vect);
  protected:
  private:
    float sigma;
    std::map<int, float> ideal_dcg_map;
};

}