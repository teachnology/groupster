import functools

import pandas as pd

from .group import Group
from .util import diversity


class Cohort:
    def __init__(self, data, groups, bools=None, nums=None):
        self.data = data.assign(
            group=pd.Series(
                [group for group, count in groups.items() for _ in range(count)]
            )
            .sample(frac=1)
            .to_list()
        )

        self.bools = bools
        self.nums = nums

    def __getitem__(self, group):
        return Group(
            data=self.data.loc[self.data.group == group, :],
            bools=self.bools,
            nums=self.nums,
        )

    @functools.cached_property
    def diversity(self):
        return diversity(data=self.data, bools=self.bools, nums=self.nums)

    def diversity_cost(self, cost_fn=None):
        return pd.Series(
            {
                group: self[group].diversity_cost(
                    cohort_diversity=self.diversity, cost_fn=cost_fn
                )
                for group in self.data.group.unique()
            }
        ).sum()

    def restriction_cost(self, keep_together=None, keep_separate=None, cost_fn=None):
        return pd.Series(
            {
                group: self[group].restriction_cost(
                    keep_together=keep_together,
                    keep_separate=keep_separate,
                    cost_fn=cost_fn,
                )
                for group in self.data.group.unique()
            }
        ).sum()

    def overview(self):
        series = [
            self.data.groupby("group").size().rename("size"),
            self.data.groupby("group")
            .apply(lambda x: self[x.name].diversity_cost(self.diversity))
            .rename("diversity_cost"),
        ]

        for col in self.bools:
            series.append(self.data.groupby("group")[col].sum().rename(col))

        for col in self.nums:
            series.append(
                self.data.groupby("group")[col]
                .agg(["mean", "std"])
                .round(2)
                .rename(lambda agg: f"{agg}({col})", axis=1)
            )

        return pd.concat(series, axis=1)
