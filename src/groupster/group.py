import re

import pandas as pd


def cost(index, cohort_val, group_val):
    if re.match(r"^bool_", index):
        return abs(cohort_val - group_val)
    elif re.match(r"^num_.*_mean$", index):
        # The difference is normalised by the cohort mean.
        return abs(cohort_val - group_val) / cohort_val
    elif re.match(r"^num_.*_std$", index):
        # The difference is normalised by the cohort mean.
        return abs(cohort_val - group_val) / cohort_val
    else:
        raise ValueError(f"Cost function does not know how to handle {index}.")


class Group:
    def __init__(self, data, bools=None, nums=None):
        self.data = data
        self.bools = bools if bools is not None else []
        self.nums = nums if nums is not None else []

    def diversity(self):
        # Bool variables
        bool_mean = {f"bool_{col}": self.data.loc[:, col].mean() for col in self.bools}

        # Means of numerical variables
        num_mean = {
            f"num_{col}_mean": self.data.loc[:, col].mean() for col in self.nums
        }

        # Standard deviations of numerical variables
        num_std = {f"num_{col}_std": self.data.loc[:, col].std() for col in self.nums}

        return pd.Series(bool_mean | num_mean | num_std)

    def cost(self, cohort_diversity, cost_fn=None):
        if cost_fn is None:
            cost_fn = cost

        return pd.Series(
            {
                idx: cost_fn(idx, cohort_diversity.loc[idx], self.diversity().loc[idx])
                for idx in cohort_diversity.index
            }
        ).sum()

    def __repr__(self):
        pass
