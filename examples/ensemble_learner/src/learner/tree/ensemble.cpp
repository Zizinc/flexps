#include "ensemble.hpp"

namespace flexps {

Ensemble::Ensemble() {
  
}

void Ensemble::add_tree(RegressionTree& tree) {
  this->tree_vect.push_back(tree);
}

}