#include "examples/ensemble_learner/src/data_loader/data_loader.hpp"
#include "examples/ensemble_learner/src/utilities/util.hpp"

#include "glog/logging.h"
#include "gtest/gtest.h"

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <cstdlib>
#include <vector>

namespace flexps {

DataLoader::DataLoader() {

}

DataLoader::DataLoader(std::string file_path) {
  this->data_format = "svm_lib";
  read_from_local_file_path(file_path);
}

DataLoader::DataLoader(std::string data_format, std::string file_path) {
  this->data_format = data_format;
  read_from_local_file_path(file_path);
}

DataLoader::DataLoader(
      std::vector<float> class_vect,
      std::vector<std::vector<float>> feat_vect_list,
      std::vector<std::map<std::string, float>> min_max_feat_list,
      int num_of_record,
      int num_of_feat,
      std::string data_format
    ) {
  this->class_vect = class_vect;
  this->feat_vect_list = feat_vect_list;
  this->min_max_feat_list = min_max_feat_list;
  this->num_of_record = num_of_record;
  this->num_of_feat = num_of_feat;
  this->data_format = data_format;
}

void DataLoader::read_hdfs_to_class_feat_vect(std::vector<std::string>& line_vect) {
  bool init_flag = false;
  this->num_of_record = 0;
  int num_of_feat = 0;

  for (std::string line: line_vect) {
    std::vector<float> val_vect = read_line_to_vect(line);
    if (init_flag == false) {
      num_of_feat = val_vect.size() - 1;
      this->feat_vect_list.resize(num_of_feat);
      init_flag = true;
    }
    this->class_vect.push_back(val_vect[0]);

    for (int i = 0; i < num_of_feat; i++) {
      this->feat_vect_list[i].push_back(val_vect[i+1]);
    }
    this->num_of_record++;
  }
}

std::vector<float> DataLoader::get_feat_vect_by_row(std::vector<std::vector<float>>& feat_vect_list, int row) {
  std::vector<float> data_row;
  for (int j = 0; j < feat_vect_list.size(); j++) {
    data_row.push_back(feat_vect_list[j][row]);
  }
  return data_row;
}

void DataLoader::split_dataset_by_feat_val(
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
      ) {
  left_feat_vect_list.resize(orig_feat_vect_list.size());
  right_feat_vect_list.resize(orig_feat_vect_list.size());
  
  for (int i = 0; i < orig_grad_vect.size(); i++) {
    if (orig_feat_vect_list[feat_id][i] < split_val) {
      left_grad_vect.push_back(orig_grad_vect[i]);
      left_hess_vect.push_back(orig_hess_vect[i]);
      for (int j = 0; j < orig_feat_vect_list.size(); j++) {
        left_feat_vect_list[j].push_back(orig_feat_vect_list[j][i]);
      }
    }
    else {
      right_grad_vect.push_back(orig_grad_vect[i]);
      right_hess_vect.push_back(orig_hess_vect[i]);
      for (int j = 0; j < orig_feat_vect_list.size(); j++) {
        right_feat_vect_list[j].push_back(orig_feat_vect_list[j][i]);
      }
    }
  }
}

std::vector<float> DataLoader::pop() {
  bool error_flag = false;
  std::vector<float> vect;

  if (class_vect.size() == 0) {
    LOG(INFO) << "Pop failed: class size is 0";
    error_flag = true;
  }
  else {
    vect.push_back(class_vect[class_vect.size() - 1]);
    class_vect.pop_back();
  }

  for (int i = 0; i < feat_vect_list.size(); i++) {
    std::vector<float> & feat_vect = feat_vect_list[i];
    if (feat_vect.size() == 0) {
      LOG(INFO) << "Pop failed: size of feat vect at column " << i << " is 0";
      error_flag = true;
    }
    else {
      vect.push_back(feat_vect[feat_vect.size() - 1]);
      feat_vect.pop_back();
    }
  }
  if (!error_flag) {
    num_of_record--;
  }
  return vect;
}

void DataLoader::push(std::vector<float>& vect) {
  if (vect.size() > 0) {
    if (class_vect.size() == 0 && feat_vect_list.size() == 0) {
      // have not been initialized yet
      int num_of_feat = vect.size() - 1;
      feat_vect_list.resize(num_of_feat);
      for (int i = 0; i < vect.size(); i++) {
        if (i == 0) {
          class_vect.push_back(vect[i]);
        }
        else {
          feat_vect_list[i-1].push_back(vect[i]);
        }
      }
      num_of_record++;
    }
    else if (vect.size() != 1 + feat_vect_list.size()) {
      LOG(INFO) << "Push failed: size of push vect and that of feat vect does not match";
    }
    else {
      for (int i = 0; i < vect.size(); i++) {
        if (i == 0) {
          class_vect.push_back(vect[i]);
        }
        else {
          feat_vect_list[i-1].push_back(vect[i]);
        }
      }
      num_of_record++;
    }
  }
  else {
    LOG(INFO) << "Push failed: vect.size() smaller than 0";
  }
}

void DataLoader::print_loaded_data() {
  for (int n = 0; n < num_of_record; n++) {
    std::stringstream ss;
    ss << class_vect[n] << " ";

    if (data_format == "svm_rank") {
      ss  << "qid:" << qid_vect[n] << " ";
    }

    for (int fid = 0; fid < feat_vect_list.size(); fid++) {
      ss << std::to_string(fid + 1) << ":" << std::to_string(feat_vect_list[fid][n]) << " ";
    }
    LOG(INFO) << ss.str();
  }
}

DataLoader DataLoader::create_dataloader_by_worker_id(int worker_id, int worker_num) {
  auto& all_feat_vect_list = this->feat_vect_list;
  auto& all_class_vect = this->class_vect;
  int data_num = all_class_vect.size();
  std::vector<std::vector<float>> resized_feat_vect_list(all_feat_vect_list.size());
  std::vector<float> resized_class_vect;
  
  int num_of_records_per_worker = data_num / worker_num;

  if (worker_id == worker_num - 1) {
    resized_class_vect.insert(resized_class_vect.end(), all_class_vect.begin() + (worker_id * num_of_records_per_worker), all_class_vect.end());
    for (int i = 0; i < all_feat_vect_list.size(); i++) {
      resized_feat_vect_list[i].insert(resized_feat_vect_list[i].end(), all_feat_vect_list[i].begin() + (worker_id * num_of_records_per_worker), all_feat_vect_list[i].end());
    }
  }
  else {
    resized_class_vect.insert(resized_class_vect.end(), all_class_vect.begin() + (worker_id * num_of_records_per_worker), 
      all_class_vect.begin() + ((worker_id + 1) * num_of_records_per_worker));
    for (int i = 0; i < all_feat_vect_list.size(); i++) {
      resized_feat_vect_list[i].insert(resized_feat_vect_list[i].end(), all_feat_vect_list[i].begin() + (worker_id * num_of_records_per_worker),
        all_feat_vect_list[i].begin() + ((worker_id + 1) * num_of_records_per_worker));
    }
  }
  DataLoader local_data_loader(
    resized_class_vect,
    resized_feat_vect_list,
    this->min_max_feat_list,
    resized_class_vect.size(), //this->num_of_record,
    this->num_of_feat,
    this->data_format
  );
  LOG(INFO) << "worker_id = " << worker_id << ", worker_num = " << worker_num 
    << ",  global data num = " << this->class_vect.size() << ", local data num = " << resized_class_vect.size();
  return local_data_loader;
}

void DataLoader::read_from_local_file_path(std::string file_path) {
  std::ifstream infile(file_path.c_str());

  std::string line;
  bool init_flag = false;
  this->num_of_record = 0;
  this->num_of_feat = 0;

  while (getline(infile, line)) {
    std::vector<float> val_vect = read_line_to_vect(line);
    if (init_flag == false) {
      num_of_feat = val_vect.size() - 1;
      this->feat_vect_list.resize(num_of_feat);
      init_flag = true;
    }
    this->class_vect.push_back(val_vect[0]);
    val_vect.erase(val_vect.begin());

    if (data_format == "svm_rank") {
      this->qid_vect.push_back(static_cast<int>(val_vect[0]));
      val_vect.erase(val_vect.begin());
    }

    for (int i = 0; i < num_of_feat; i++) {
      this->feat_vect_list[i].push_back(val_vect[i]);
    }

    this->num_of_record++;
  }
  ASSERT_NE(this->num_of_record, 0);
}

std::vector<float> DataLoader::read_line_to_vect(std::string line) {
  // line format (svm_lib): <class_val>  1:<feat_val> 2:<feat_val> ...

  std::vector<float> vect;
  std::vector<std::string> feat_element_vect = split(line, "  ");

  float class_val = atof(feat_element_vect[0].c_str());
  vect.push_back(class_val);

  feat_element_vect.erase(feat_element_vect.begin());
  
  if (this->data_format == "svm_rank") {
    if (split(feat_element_vect[0], ":")[0] == "qid") {
      vect.push_back(atof(split(feat_element_vect[0], ":")[1].c_str()));
      feat_element_vect.erase(feat_element_vect.begin());
    }
  }

  for (std::string element: feat_element_vect) {
    float feat_val = atof(split(element, ":")[1].c_str());
    vect.push_back(feat_val);
  }

  return vect;
}

}