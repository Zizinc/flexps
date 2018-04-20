#pragma once

#include "LEARNER_TYPE.hpp"
#include "learner.hpp"

namespace flexps {

class LearnerFactory {
  public:
    LearnerFactory();
	Learner *create_learner(LearnerType learner_type);
};

}