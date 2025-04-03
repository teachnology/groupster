import re


def diversity_cost(index, cohort_val, group_val):
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


def restriction_cost(data, keep_together=None, keep_separate=None):
    cost = 0

    if keep_together is not None:
        for i in keep_together:
            cost -= data.index.isin(i).sum() ** 2  # negative cost if people are together

    if keep_separate is not None:
        for i in keep_separate:
            cost += data.index.isin(i).sum() ** 2  # positive cost if people are not separate

    return cost
