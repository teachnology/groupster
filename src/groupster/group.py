import pandas as pd
from .util import diversity_cost, restriction_cost, diversity


class Group:
    def __init__(self, data, bools=None, nums=None):
        self.data = data
        self.bools = bools
        self.nums = nums

    @property
    def diversity(self):
        return diversity(data=self.data, bools=self.bools, nums=self.nums)

    def diversity_cost(self, cohort_diversity, cost_fn=None):
        if cost_fn is None:
            cost_fn = diversity_cost

        return cost_fn(
            cohort_diversity=cohort_diversity,
            group_diversity=self.diversity,
        )

    def restriction_cost(self, keep_together=None, keep_separate=None, cost_fn=None):
        if cost_fn is None:
            cost_fn = restriction_cost

        return cost_fn(self.data, keep_together, keep_separate)
