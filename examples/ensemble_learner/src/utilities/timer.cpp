#include <algorithm>

#include "timer.hpp"

#include "glog/logging.h"

namespace flexps {

Timer::Timer() {
}

void Timer::start_clock(std::string clock_name) {
  this->time_point_map[clock_name] = std::chrono::high_resolution_clock::now();
}

void Timer::add_time(std::string clock_name) {
  float elapse_time = 0.0;

  if (this->time_point_map.count(clock_name)) {
  	elapse_time = std::chrono::duration<double, std::milli>(std::chrono::high_resolution_clock::now() 
  	  - this->time_point_map[clock_name]).count() / 1000;
  }

  if (this->time_map.count(clock_name)) {
  	this->time_map[clock_name] += elapse_time;
  }
  else {
  	this->time_map[clock_name] = elapse_time;
  }
}

float Timer::get_time(std::string clock_name) {
  float time = 0.0;
  if (this->time_map.count(clock_name)) {
  	time = this->time_map[clock_name];
  }
  return time;
}

}
