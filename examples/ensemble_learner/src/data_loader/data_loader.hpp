#pragma once

#include <map>
#include <string>
#include <vector>

namespace flexps {

class DataLoader {
  public:
    DataLoader();
    DataLoader(std::string file_path);
    DataLoader(std::string data_format, std::string file_path);
    DataLoader(
      std::vector<float> class_vect,
      std::vector<std::vector<float>> feat_vect_list,
	  std::vector<std::map<std::string, float>> min_max_feat_list,
      int num_of_record,
      int num_of_feat,
      std::string data_format
    );
    void read_hdfs_to_class_feat_vect(std::vector<std::string>& line_vect);
    // Getter and setter ++
    void set_class_vect(std::vector<float>& class_vect) { this->class_vect = class_vect; }
    std::vector<float>& get_class_vect() { return this->class_vect; }
    void set_qid_vect(std::vector<int> qid_vect) { this->qid_vect = qid_vect; }
    std::vector<int>& get_qid_vect() { return this->qid_vect; }
    void set_feat_vect_list(std::vector<std::vector<float>>& feat_vect_list) { this->feat_vect_list = feat_vect_list; }
    std::vector<std::vector<float>>& get_feat_vect_list() { return this->feat_vect_list; }
    void set_min_max_feat_list(std::vector<std::map<std::string, float>>& min_max_feat_list) { this->min_max_feat_list = min_max_feat_list; }
    std::vector<std::map<std::string, float>>& get_min_max_feat_list() { return this->min_max_feat_list; }
    void set_num_of_record(int num_of_record) { this->num_of_record = num_of_record; }
    int get_num_of_record() { return this->num_of_record; }
    void set_num_of_feat(int num_of_feat) { this->num_of_feat = num_of_feat; }
    int get_num_of_feat() { return this->num_of_feat; }
    void set_data_format(std::string data_format) { this->data_format = data_format; }
    std::string get_data_format() { return this->data_format; }
    // Getter and setter --
    static std::vector<float> get_feat_vect_by_row(std::vector<std::vector<float>>& feat_vect_list, int row);
    static void split_dataset_by_feat_val(
      int feat_id,
      float split_val,
      std::vector<float>& orig_grad_vect,
      std::vector<float>& left_grad_vect,
      std::vector<float>& right_grad_vect,
      std::vector<float>& orig_hess_vect,
      std::vector<float>& left_hess_vect,
      std::vector<float>& right_hess_vect,
      std::vector<std::vector<float>>& orig_feat_vect_list,
      std::vector<std::vector<float>>& left_feat_vect_list,
      std::vector<std::vector<float>>& right_feat_vect_list
      );
    std::vector<float> pop();
    void push(std::vector<float>& vect);
    void print_loaded_data();
    DataLoader create_dataloader_by_worker_id(int worker_id, int worker_num);
  protected:
    void read_from_local_file_path(std::string file_path);
    std::vector<float> read_line_to_vect(std::string line);
  private:
    std::vector<float> class_vect;
    std::vector<int> qid_vect;
    std::vector<std::vector<float>> feat_vect_list;
	std::vector<std::map<std::string, float>> min_max_feat_list;
    int num_of_record;
    int num_of_feat;
    std::string data_format;
};

}