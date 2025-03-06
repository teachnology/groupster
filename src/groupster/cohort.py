import pandas as pd
import random


# bools and nums should be dics with weights

class Cohort:
    def __init__(self, data, groups, bools=None, nums=None):
        group_col = [group for group, count in groups.items() for _ in range(count)]
        data = data.assign(
            group=lambda df_: random.sample(
                [group for group, count in groups.items() for _ in range(count)],
                k=len(df_),
            )
        ).astype({"group": "category"})

        self.data = data
        self.bools = list(bools) if bools is not None else []
        self.nums = list(nums) if nums is not None else []
        # self.groups = groups

    def diversity(self):
        # Bool variables
        bool_mean = {col: self.data.loc[:, col].mean() for col in self.bools}

        # Means of numerical variables
        num_mean = {f'{col}_mean': self.data.loc[:, col].mean() for col in self.nums}

        # Standard deviations of numerical variables
        num_std = {f'{col}_std': self.data.loc[:, col].std() for col in self.nums}

        return pd.Series(bool_mean | num_mean | num_std)

    def __getitem__(self, key):
        return self.data.loc[self.data.group == key, :]
