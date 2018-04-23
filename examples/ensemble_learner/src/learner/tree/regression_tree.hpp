#pragma once

#include "examples/ensemble_learner/src/learner/LEARNER_TYPE.hpp"

#include <map>
#include <string>
#include <vector>
#include "worker/kv_client_table.hpp"

namespace flexps {

class RegressionTree {
  public:
    RegressionTree();
    void init(
      LearnerType learner_type,
      std::vector<std::vector<float>>& feat_vect_list, 
      std::vector<std::map<std::string, float>>& min_max_feat_list, 
      std::vector<float>& grad_vect, 
      std::vector<float>& hess_vect,
      std::map<std::string, float>& params
    );
    void set_kv_tables(std::map<std::string, std::unique_ptr<KVClientTable<float>>>* kv_tables);
    void train();
    std::vector<std::vector<float>> find_candidate_splits();
    void find_best_candidate_split(std::vector<std::vector<float>>& candidate_split_vect_list);
    void find_predict_val();
    void reset_kv_tables();
    void train_child();
    float predict(std::vector<float>& vect);
    void update_leafs(float factor);
    // Getter and setter ++
    std::map<std::string, std::vector<float>> get_additional_vect_map() { return this->additional_vect_map; }
    void set_additional_vect_map(std::map<std::string, std::vector<float>> additional_vect_map) { this->additional_vect_map = additional_vect_map; }
    // Getter and setter --
  protected:
    std::vector<Key> push_quantile_sketch(int& ps_key_ptr, std::vector<float>& feat_vect, std::map<std::string, float>& min_max_feat, std::vector<float>& _push_val_vect);
    std::vector<float> find_candidate_split(std::vector<float>& sketch_hist_vect, std::map<std::string, float>& min_max_feat);
    std::vector<Key> push_local_grad_hess(int& ps_key_ptr, std::vector<float>& feat_vect, std::vector<float>& candidate_split_vect
      , std::vector<float>& grad_vect, std::vector<float>& hess_vect, std::vector<float>& _push_val_vect);
    std::map<std::string, float> find_best_split(std::vector<float>& grad_hess_vect, float complexity_of_leaf);
	std::vector<Key> push_local_grad_sum(int& ps_key_ptr, std::vector<float>& _push_val_vect);
    bool check_to_stop();
    void set_depth(int depth);
    void split_and_set_additional_vect_map();
  private:
    LearnerType learner_type;
    std::vector<std::vector<float>> feat_vect_list;
    std::vector<std::map<std::string, float>> min_max_feat_list;
    std::vector<float> grad_vect; 
    std::vector<float> hess_vect;
    std::map<std::string, std::unique_ptr<KVClientTable<float>>>* kv_tables;
    
    // Additional vector values
    std::map<std::string, std::vector<float>> additional_vect_map;

    RegressionTree* left_child;
    RegressionTree* right_child;
    int depth;
    float predict_val;
    bool is_leaf;

    int feat_id;
    float split_val;
    float max_gain;

    // stop criteria
    std::map<std::string, float> params;
    
    // ps key set
    std::vector<std::vector<Key>> quantile_sketch_key_vect_list;
    std::vector<std::vector<Key>> grad_hess_key_vect_list;
    // the first is sum, the second is num
    std::vector<Key> local_grad_sum_key_vect_list;

    // ps val set (for reset propose)
    std::vector<std::vector<float>> quantile_sketch_val_vect_list;
    std::vector<std::vector<float>> grad_hess_val_vect_list;
    std::vector<float> local_grad_sum_val_vect_list;
};

}