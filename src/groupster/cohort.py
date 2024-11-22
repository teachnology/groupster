import pandas as pd


class Cohort:
    def __init__(self, data, bools=None, nums=None):
        self.data = data
        self.bools = bools
        self.nums = nums
