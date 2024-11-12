import pandas as pd


class Cohort:
    def __init__(self, df, categorical_columns, numerical_columns):
        self.df = df
        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
