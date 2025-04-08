import re

import pandas as pd


def diversity(data, bools=None, nums=None):
    bool_mean = (
        {f"bool_{col}": data.loc[:, col].mean() for col in bools}
        if bools is not None
        else {}
    )

    # Means of numerical variables
    num_mean = (
        {f"num_{col}_mean": data.loc[:, col].mean() for col in nums}
        if nums is not None
        else {}
    )

    # Standard deviations of numerical variables
    num_std = (
        {f"num_{col}_std": data.loc[:, col].std() for col in nums}
        if nums is not None
        else {}
    )

    return pd.Series(bool_mean | num_mean | num_std)


def diversity_cost(cohort_diversity, group_diversity):
    cost = 0

    for idx in cohort_diversity.index:
        cohort_val = cohort_diversity.loc[idx]
        group_val = group_diversity.loc[idx]

        if re.match(r"^bool_", idx):
            cost += abs(cohort_val - group_val)
        elif re.match(r"^num_.*_mean$", idx):
            # The difference is normalised by the cohort mean.
            cost += abs(cohort_val - group_val) / cohort_val
        elif re.match(r"^num_.*_std$", idx):
            # The difference is normalised by the cohort mean of the same variable.
            variable_name = re.match(r"^num_(.*)_std$", idx).group(1)
            cohort_mean = cohort_diversity.loc[f"num_{variable_name}_mean"]
            cost += abs(cohort_val - group_val) / cohort_mean
        else:
            raise ValueError(f"Cost function does not know how to handle {idx}.")

    return cost


def restriction_cost(data, keep_together=None, keep_separate=None, bool_min=None):
    cost = 0

    if keep_together is not None:
        for i in keep_together:
            cost -= (
                data.index.isin(i).sum() ** 2  # square to add nonlinearity
            )  # negative cost if people are together

    if keep_separate is not None:
        for i in keep_separate:
            cost += (
                data.index.isin(i).sum() ** 2  # square to add nonlinearity
            )  # positive cost if people are not separate

    if bool_min is not None:
        for name, n in bool_min.items():
            cost += (data[name].sum() < n) * 10

    return cost
