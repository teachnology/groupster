import random
import pandas as pd
from .group import Group

# bools and nums should be dics with weights


class Cohort(Group):
    def __init__(self, data, groups, bools=None, nums=None):
        self.data = data.assign(
            group=lambda df_: random.sample(
                [group for group, count in groups.items() for _ in range(count)],
                k=len(df_),
            )
        ).astype({"group": "category"})

        self.bools = list(bools) if bools is not None else []
        self.nums = list(nums) if nums is not None else []

    def __getitem__(self, group):
        return Group(
            data=self.data.loc[self.data.group == group, :],
            bools=self.bools,
            nums=self.nums,
        )

    def diversity_cost(self, cost_fn=None):
        return sum(
            self[group].diversity_cost(
                cohort_diversity=self.diversity(), cost_fn=cost_fn
            )
            for group in self.data.group.unique()
        )

    def restriction_cost(self, keep_together=None, keep_separate=None, cost_fn=None):
        return sum(
            self[group].restriction_cost(
                keep_together=keep_together,
                keep_separate=keep_separate,
                cost_fn=cost_fn,
            )
            for group in self.data.group.unique()
        )

    def overview(self):
        gb = self.data.groupby("group")
        series = [
            gb.size().rename("size"),
            gb.apply(lambda x: self[x.name].diversity_cost(self.diversity())).rename("diversity_cost"),
        ]

        for col in self.bools:
            series.append(gb[col].sum().rename(col))

        for col in self.nums:
            series.append(
                gb[col]
                .agg(["mean", "std"])
                .round(2)
                .rename(lambda agg: f"{agg}({col})", axis=1)
            )

        return pd.concat(series, axis=1)
