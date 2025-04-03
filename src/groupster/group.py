import pandas as pd
from .util import diversity_cost, restriction_cost


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

    def diversity_cost(self, cohort_diversity, cost_fn=None):
        if cost_fn is None:
            cost_fn = diversity_cost

        return pd.Series(
            {
                idx: cost_fn(idx, cohort_diversity.loc[idx], self.diversity().loc[idx])
                for idx in cohort_diversity.index
            }
        ).sum()

    def restriction_cost(self, keep_together=None, keep_separate=None, cost_fn=None):
        if cost_fn is None:
            cost_fn = restriction_cost

        return cost_fn(self.data, keep_together, keep_separate)