#include "learner_factory.hpp"
#include "examples/ensemble_learner/src/learner/tree/dart.hpp"
#include "examples/ensemble_learner/src/learner/tree/gbdt.hpp"
#include "examples/ensemble_learner/src/learner/tree/lambda_mart.hpp"

#include "glog/logging.h"

namespace flexps {

LearnerFactory::LearnerFactory() {
  
}

Learner *LearnerFactory::create_learner(LearnerType learner_type) {
  Learner *learner;
  
  switch (learner_type) {
    case GBDT_LEARNER:
      learner = new GBDT;
      break;
    case LAMBDAMART_LEARNER:
      learner = new LambdaMART;
      break;
    case DART_LEARNER:
      learner = new DART;
      break;
    default:
      LOG(INFO) << "Unknown learner type";
      exit(-1);
      break;
  }
  return learner;
}

}