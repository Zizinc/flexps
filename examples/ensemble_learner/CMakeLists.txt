include_directories(${PROJECT_SOURCE_DIR} ${HUSKY_EXTERNAL_INCLUDE})

file(GLOB ensemble-learner-src-files
  src/channel/channel_util.cpp
  src/data_loader/data_loader.cpp
  src/learner/learner.cpp
  src/learner/learner_factory.cpp
  src/learner/tree/dart.cpp
  src/learner/tree/ensemble.cpp
  src/learner/tree/gbdt.cpp
  src/learner/tree/lambda_mart.cpp
  src/learner/tree/regression_tree.cpp
  src/metric/ndcg_scorer.cpp
  src/metric/err_scorer.cpp
  src/parameter_server/parameter_server.cpp
  src/utilities/math_tools.cpp
  src/utilities/timer.cpp
  )
add_library(ensemble-learner-objs OBJECT ${ensemble-learner-src-files})
set_property(TARGET ensemble-learner-objs PROPERTY CXX_STANDARD 11)
add_dependencies(ensemble-learner-objs ${external_project_dependencies})

# GBDT Example
add_executable(GBDTExample gbdt_example.cpp)
target_link_libraries(GBDTExample flexps)
target_link_libraries(GBDTExample ${HUSKY_EXTERNAL_LIB})
target_link_libraries(GBDTExample ${GTEST_LIBRARIES})
set_property(TARGET GBDTExample PROPERTY CXX_STANDARD 11)
add_dependencies(GBDTExample ${external_project_dependencies})

# DART Example
add_executable(DARTExample dart_example.cpp)
target_link_libraries(DARTExample flexps)
target_link_libraries(DARTExample ${HUSKY_EXTERNAL_LIB})
target_link_libraries(DARTExample ${GTEST_LIBRARIES})
set_property(TARGET DARTExample PROPERTY CXX_STANDARD 11)
add_dependencies(DARTExample ${external_project_dependencies})