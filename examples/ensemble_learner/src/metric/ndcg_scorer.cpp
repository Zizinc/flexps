#include "ndcg_scorer.hpp"

#include "glog/logging.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <cmath>
#include <math.h>
#include <string>
#include <vector>

#include "examples/ensemble_learner/src/utilities/util.hpp"

namespace flexps {

NDCGScorer::NDCGScorer() {
  this->sigma = 1.0;
}

float NDCGScorer::calculate_dcg(std::vector<float> vect) {
  float dcg = 0.0;

  int size = vect.size() > k? k: vect.size();
  for (int i = 0; i < size; i++) {
    dcg += (pow(2.0, vect[i]) - 1.0) / (log2((i + 1.0) + 1.0));
  }

  return dcg;
}

float NDCGScorer::calculate_ideal_dcg(std::vector<float> vect) {
  // Sort vect in descending order
  std::sort(vect.begin(), vect.end(), std::greater<float>());
  return calculate_dcg(vect);
}

float NDCGScorer::get_ideal_dcg(int qid, std::vector<float> vect) {
  float ideal_dcg = 0.0;
  if (this->ideal_dcg_map.find(qid) == this->ideal_dcg_map.end()) {
    update_ideal_dcg_map(qid, vect);
  }
  ideal_dcg = this->ideal_dcg_map[qid];
  return ideal_dcg;
}

float NDCGScorer::calculate_ndcg(int qid, std::vector<float> vect) {
  float ndcg = 0.0;
  float dcg = calculate_dcg(vect);
  float ideal_dcg = get_ideal_dcg(qid, vect);
  if (std::abs(ideal_dcg) > 0) {
    ndcg = dcg / ideal_dcg;
  }
  return ndcg;
}

float NDCGScorer::calculate_delta_ndcg(int qid, std::vector<float> vect, int position_i, int position_j) {
  int max_idx = vect.size() > this->k? this->k: vect.size();
  max_idx -= 1;

  if (position_i > max_idx || position_j > max_idx) {
    return 0.0;
  }
  // Use simplied formula instead of the formal one
  float ideal_dcg = get_ideal_dcg(qid, vect);

  float delta_ndcg = (pow(2.0, vect[position_i]) - pow(2.0, vect[position_j])) 
      * ( (1.0 / log2((position_i + 1.0) + 1.0)) - (1.0 / log2((position_j + 1.0) + 1.0)) );
  delta_ndcg = delta_ndcg / ideal_dcg;
  return std::abs(delta_ndcg);
}

float NDCGScorer::calculate_rho(std::vector<float> vect, std::vector<float> score_vect, int position_i, int position_j) {
  float rho = 1.0 / (1.0 + exp(this->sigma * (score_vect[position_i] - score_vect[position_j])));
  return rho;
}

float NDCGScorer::calculate_lambda(int qid, std::vector<float> vect, std::vector<float> score_vect, int position_i, int position_j) {
  float delta_ndcg = calculate_delta_ndcg(qid, vect, position_i, position_j);
  float rho = calculate_rho(vect, score_vect, position_i, position_j);
  return rho * std::abs(delta_ndcg);
}

std::map<std::string, std::vector<float>> NDCGScorer::calculate_lambda_and_weight_vect(std::vector<float> vect, std::vector<float> score_vect) {
  std::map<std::string, std::vector<float>> result_map;
  std::vector<float> lambda_vect(vect.size(), 0.0);
  std::vector<float> weight_vect(vect.size(), 0.0);

  EXPECT_NE(this->qid_map.size(), 0);
  for (auto& kv_pair: this->qid_map) {
    int qid = kv_pair.first;
    auto& idx_vect = kv_pair.second;

    // Retrieve records with the same qid
    std::vector<float> tmp_vect;
    std::vector<float> tmp_score_vect;
    for (auto& idx: idx_vect) {
      tmp_vect.push_back(vect[idx]);
      tmp_score_vect.push_back(score_vect[idx]);
    }
    std::map<std::string, std::vector<float>> tmp_result_map = calculate_tmp_lambda_and_weight_vect(qid, tmp_vect, tmp_score_vect);
    std::vector<float> tmp_lambda_vect = tmp_result_map["lambda_vect"];
    std::vector<float> tmp_weight_vect = tmp_result_map["weight_vect"];
    for (int i = 0; i < tmp_lambda_vect.size(); i++) {
      lambda_vect[idx_vect[i]] = tmp_lambda_vect[i];
      weight_vect[idx_vect[i]] = tmp_weight_vect[i];
    }
    result_map["lambda_vect"] = lambda_vect;
    result_map["weight_vect"] = weight_vect;
  }
  return result_map;
}

std::map<std::string, std::vector<float>> NDCGScorer::calculate_tmp_lambda_and_weight_vect(int qid, std::vector<float> vect, std::vector<float> score_vect) {
  std::vector<float> lambda_vect(vect.size(), 0.0);
  std::vector<float> weight_vect(vect.size(), 0.0);
  for (int i = 0; i < vect.size(); i++) {
    for (int j = 0; j < vect.size(); j++) {
      if (i == j) { continue; }

  	  if (vect[i] > vect[j]) {
        float delta_ndcg = calculate_delta_ndcg(qid, vect, i, j);
        float rho = calculate_rho(vect, score_vect, i, j);
        float lambda_ij = rho * delta_ndcg;

        lambda_vect[i] += lambda_ij;
        lambda_vect[j] -= lambda_ij;

        weight_vect[i] += rho * (1.0 - rho) * delta_ndcg;
        weight_vect[j] += rho * (1.0 - rho) * delta_ndcg;
      }
    }
  }

  std::map<std::string, std::vector<float>> result_map;
  result_map["lambda_vect"] = lambda_vect;
  result_map["weight_vect"] = weight_vect;
  return result_map;
}

void NDCGScorer::update_ideal_dcg_map(int qid, std::vector<float> vect) {
  float ideal_dcg = calculate_ideal_dcg(vect);
  this->ideal_dcg_map[qid] = ideal_dcg;
}

float NDCGScorer::calculate_score(int qid, std::vector<float> test_vect, std::vector<float> class_vect) {
  std::vector<int> index_vect(test_vect.size(), 0.0);
  
  // Find the sorted sequence (desc)
  for (int i = 0; i < index_vect.size(); i++) {
    index_vect[i] = i;
  }
  sort(index_vect.begin(), index_vect.end(),
    [&](const int& a, const int& b) {
        return (test_vect[a] > test_vect[b]);
    }
  );

  // Order class_vect in that sorted sequence
  std::vector<float> reordered_class_vect(class_vect.size(), 0.0);
  EXPECT_EQ(test_vect.size(), class_vect.size());
  for (int i = 0; i < index_vect.size(); i++) {
    reordered_class_vect[i] = class_vect[index_vect[i]];
  }
  float score = calculate_ndcg(qid, reordered_class_vect);
  return score;
}

float NDCGScorer::calculate_score(std::vector<float> test_vect, std::vector<float> class_vect) {
  float score = 0.0;

  for (auto& kv_pair: this->qid_map) {
    int qid = kv_pair.first;
    auto& idx_vect = kv_pair.second;

    // Retrieve records with the same qid
    std::vector<float> tmp_test_vect;
    std::vector<float> tmp_class_vect;
    for (auto& idx: idx_vect) {
      tmp_test_vect.push_back(test_vect[idx]);
      tmp_class_vect.push_back(class_vect[idx]);
    }
    score += calculate_score(qid, tmp_test_vect, tmp_class_vect);
  }

  return score;
}

}