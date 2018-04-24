#include "glog/logging.h"
#include "gtest/gtest.h"

#include "examples/ensemble_learner/src/utilities/timer.hpp"

#include <thread>

namespace flexps {
namespace {

class TestTimer: public testing::Test {
 public:
  TestTimer() {}
  ~TestTimer() {}

 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(TestTimer, SimpleRun) {
  Timer timer;
  timer.start_clock("my_clock");
  std::this_thread::sleep_for(std::chrono::milliseconds(5000));
  timer.add_time("my_clock");
  float time = timer.get_time("my_clock");
  LOG(INFO) << "time from my_clock is " << time;

  timer.start_clock("my_clock");
  std::this_thread::sleep_for(std::chrono::milliseconds(6000));
  timer.add_time("my_clock");
  time = timer.get_time("my_clock");
  LOG(INFO) << "time from my_clock is " << time;
}

}
}