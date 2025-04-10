import re

import pandas as pd


def diversity(data, bools=None, nums=None):
    """Compute the diversity of cohort in ``data``.

    For each boolean variable, the mean is computed. For each numerical variable,
    the mean and standard deviation are computed.
    The result is a pandas Series with the following format:
    - ``bool_<name>``: mean of the boolean variable
    - ``num_<name>_mean``: mean of the numerical variable
    - ``num_<name>_std``: standard deviation of the numerical variable

    Parameters
    ----------
    data : pd.DataFrame
        The data to compute the diversity of.
    bools : list of str, optional
        The names of the boolean variables to compute the diversity of.
    nums : list of str, optional
        The names of the numerical variables to compute the diversity of.

    Returns
    -------
    pd.Series
        The diversity of the cohort in ``data``. The index is the name of the variable,
        and the value is the mean or standard deviation of the variable.

    """
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
    """Compute the diversity cost of a group.

    The cost is computed as the sum of costs for each variable in the diversity. The
    total cost of the group is the sum of:
    - The absolute difference between the cohort and group diversity for boolean
      variables
    - The absolute difference between the cohort and group diversity means for numerical
      variables, normalised by the cohort mean for that variable.
    - The absolute difference between the cohort and group diversity standard deviations
      for numerical variables, normalised by the cohort mean for that variable.

    Parameters
    ----------
    cohort_diversity : pd.Series
        The diversity of the cohort.
    group_diversity : pd.Series
        The diversity of the group.

    Returns
    -------
    float
        The diversity cost of the group.

    """
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
    """Compute the restriction cost of a group.

    The cost is computed as the sum of costs for each restriction. The total cost of the
    group is the sum of:
    - The number of people that are actually together squared(negative cost)
    - The number of people that are actually separate squared (positive cost)
    - The number of people that are below the minimum for a boolean variable (positive
      cost) times 10

    Parameters
    ----------
    keep_together : list of list of str, optional
            A list of lists of indices in ``data`` that should be kept together.
    keep_separate : list of list of str, optional
        A list of lists of indices in ``data`` that should be kept separate.
    bool_min : float, optional
        The minimum number of people with the boolean characteristic in the group.
        For example, if we want at least two females in each group, we can set
        ``bool_min={'female': 2}``.

    Returns
    -------
    float
        The restriction cost of the group.

    """
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
