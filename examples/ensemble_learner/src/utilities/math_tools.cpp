#include "math_tools.hpp"

#include "glog/logging.h"

#include <algorithm>
#include <limits>
#include <math.h>
#include <string>
#include <vector>

namespace flexps {

float calculate_gradient(float actual, float predict, std::string loss_function, int order) {
  float result;
  switch(order) {
    case 1: // first order
      if (loss_function == "square_error") {
        result = actual - predict;
      }
      else {
        result = 0;
      }
      break;
    case 2: //second order
      if (loss_function == "square_error") {
        result = 1;
      }
      else {
        result = 0;
      }
      break;
    default:
      result = 0;
      break;

  }
  return result;
}

std::map<std::string, float> find_min_max(std::vector<float> vect) {
  float min = std::numeric_limits<float>::infinity();
  float max = -std::numeric_limits<float>::infinity();

  for (float val: vect) {
    if (val < min) {
      min = val;
    }
    else if (val > max) {
      max = val;
    }
  }
  
  std::map<std::string, float> res;
  res["min"] = min;
  res["max"] = max;

  return res;
}

float calculate_rmse(std::vector<float> vect1, std::vector<float> vect2) {
  float rmse = 0.0;

  for (int i = 0; i < vect1.size(); i++) {
    
    float error = vect1[i] - vect2[i];
    rmse += error * error;
  }
  rmse = rmse / vect1.size();
  rmse = sqrt(rmse);

  return rmse;
}

float calculate_rmse(float & SSE, float & NUM) {
  return sqrt(SSE / NUM);
}

void get_SSE_and_NUM(std::vector<float> vect1, std::vector<float> vect2, float & SSE, int & NUM) {
  float _SSE = 0.0;

  for (int i = 0; i < vect1.size(); i++) {
    
    float error = vect1[i] - vect2[i];
    _SSE += error * error;
  }
  
  SSE = _SSE;
  NUM = vect1.size();

  return;
}

}