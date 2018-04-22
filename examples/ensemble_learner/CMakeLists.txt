include_directories(${PROJECT_SOURCE_DIR} ${HUSKY_EXTERNAL_INCLUDE})

file(GLOB ensemble-learner-src-files
  src/channel/channel_util.cpp
  src/data_loader/data_loader.cpp
  src/learner/learner.cpp
  src/learner/learner_factory.cpp
  src/learner/tree/ensemble.cpp
  src/learner/tree/gbdt.cpp
  src/learner/tree/lambda_mart.cpp
  src/learner/tree/regression_tree.cpp
  src/metric/ndcg_scorer.cpp
  src/metric/err_scorer.cpp
  src/parameter_server/parameter_server.cpp
  src/utilities/math_tools.cpp
  )
add_library(ensemble-learner-objs OBJECT ${ensemble-learner-src-files})
set_property(TARGET ensemble-learner-objs PROPERTY CXX_STANDARD 11)
add_dependencies(ensemble-learner-objs ${external_project_dependencies})
