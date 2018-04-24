#pragma once

#include <chrono>
#include <map>
#include <string>

namespace flexps {

class Timer {
  public:
    Timer();
    void start_clock(std::string clock_name);
    void add_time(std::string clock_name);
    float get_time(std::string clock_name);
  private:
    std::map<std::string, float> time_map;
    std::map<std::string, std::chrono::high_resolution_clock::time_point> time_point_map;
};

}