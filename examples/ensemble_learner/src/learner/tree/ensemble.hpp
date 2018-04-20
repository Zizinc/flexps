#pragma once

#include "regression_tree.hpp"

#include <vector>

namespace flexps {

class Ensemble {
  public:
    Ensemble();
	void add_tree(RegressionTree& tree);
	std::vector<RegressionTree> get_trees() { return this->tree_vect; }
	std::vector<RegressionTree>::iterator begin() { return tree_vect.begin(); }
	std::vector<RegressionTree>::iterator end() { return tree_vect.end(); }
  private:
    std::vector<RegressionTree> tree_vect;
};

}