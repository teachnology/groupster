import functools

import pandas as pd

from .group import Group
from .util import diversity


class Cohort:
    """A cohort of people, each assigned to a group.

    The cohort data is ``pd.DataFrame`` with a ``group`` column, and the groups are
    assigned randomly at initialisation. The number of people in each group is specified
    by the groups dictionary. For computing the diversity cost, the ``bools`` and
    ``nums`` columns are specified. The bools columns are boolean columns (e.g.
    categorical characteristics), and the nums columns are numeric (continuous values)
    columns.

    Parameters
    ----------
    data : pd.DataFrame
        The cohort data. Index must be unique.
    groups : dict
        A dictionary with the group names as keys and the number of people in each group
        as values.
    bools : list, optional
        A list of boolean columns in the data. These are categorical characteristics.
    nums : list, optional
        A list of numeric columns in the data. These are continuous values.

    """

    def __init__(self, data, groups, bools=None, nums=None):
        self.data = data.assign(
            group=pd.Series(
                [group for group, count in groups.items() for _ in range(count)]  # check np.repeat
            )
            .sample(frac=1)
            .to_list()
        )

        self.bools = bools
        self.nums = nums

    def __getitem__(self, group):
        """Extract a group from the cohort.

        Parameters
        ----------
        group : str
            The name of the group.

        Returns
        -------
        Group
            The group object.

        """
        return Group(
            data=self.data.loc[self.data.group == group, :],
            bools=self.bools,
            nums=self.nums,
        )

    @functools.cached_property
    def diversity(self):
        """Compute the diversity of the cohort.

        This is a cached property, so it is only computed once and then stored.
        Modifying the ``data`` attribute directly might result in inconsistent results.

        For more details, see the `util.diversity` function.

        """
        return diversity(data=self.data, bools=self.bools, nums=self.nums)

    def diversity_cost(self, cost_fn=None):
        """Compute the diversity cost of the cohort.

        The diversity cost is the sum of the diversity costs of each group. For more
        information, see ``Group.diversity_cost``.

        Parameters
        ----------
        cost_fn : callable, optional
            A function to compute the group cost.

        Returns
        -------
        float
            The diversity cost of the cohort.

        """
        return pd.Series(
            {
                group: self[group].diversity_cost(
                    cohort_diversity=self.diversity, cost_fn=cost_fn
                )
                for group in self.data.group.unique()
            }
        ).sum()

    def restriction_cost(
        self, keep_together=None, keep_separate=None, bool_min=None, cost_fn=None
    ):
        """Compute the restriction cost of the cohort.

        The restriction cost is the sum of the restriction costs of each group. For more
        information, see ``Group.restriction_cost``.

        Parameters
        ----------
        keep_together : list of list of str, optional
            A list of lists of indices in ``data`` that should be kept together.
        keep_separate : list of list of str, optional
            A list of list of indices in ``data`` that should be kept separate.
        bool_min : float, optional
            The minimum number of people with the boolean characteristic in the group.
            For example, if we want at least two females in each group, we can set
            ``bool_min={'female': 2}``.
        cost_fn : callable, optional
            A function to compute the restriction cost of a group.

        Returns
        -------
        float
            The restriction cost of the cohort.

        """
        return pd.Series(
            {
                group: self[group].restriction_cost(
                    keep_together=keep_together,
                    keep_separate=keep_separate,
                    bool_min=bool_min,
                    cost_fn=cost_fn,
                )
                for group in self.data.group.unique()
            }
        ).sum()

    def overview(self):
        """Compute the overview DataFrame of the cohort.

        The overview is a ``pd.DataFrame`` with the following columns:
        - ``size``: the number of people in each group
        - ``diversity_cost``: the diversity cost of each group
        - ``<bool_name>``: the sum of the boolean variable in each group
        - ``mean(<num_name>)``: the mean of the numerical variable in each group
        - ``std(<num_name>)``: the standard deviation of the numerical variable in each
          group

        Returns
        -------
        pd.DataFrame
            The overview of the cohort. The index is the group name, and the columns are
            the size, diversity cost, and the boolean and numerical variables.

        """
        series = [
            self.data.groupby("group").size().rename("size"),
            self.data.groupby("group")
            .apply(
                lambda x: self[x.name].diversity_cost(self.diversity),
                include_groups=False,
            )
            .rename("diversity_cost"),
        ]

        if self.bools is not None:
            for col in self.bools:
                series.append(self.data.groupby("group")[col].sum().rename(col))

        if self.nums is not None:
            for col in self.nums:
                series.append(
                    self.data.groupby("group")[col]
                    .agg(["mean", "std"])
                    .round(2)
                    .rename(lambda agg: f"{agg}({col})", axis=1)
                )

        return pd.concat(series, axis=1)
