import collections
import collections.abc
import functools
import random

import pandas as pd


class TeamBuilder:
    """Class for building diverse teams.

    Parameters
    ----------
    data: pandas.DataFrame

        DataFrame with details about individuals.

    identifier: str

            Column name in ``data`` used to specify individuals.

    groups: Iterable(int)

        The number of people per group.

    categorical: Iterable(str)
        Column names of variables that should be trated as categorical.
    continuous: Iterable(str)
        Column names of variables that should be trated as continuous.
    together: Iterable(Iterable(str))
        Groups of individuals that should be kept together.
    separate: Iterable(Iterable(str))
        Groups of individuals that should be kept separate.
    minimum_values: dict

        Dictionary of required minimum members with the same categorical value
        in each group.

    """

    def __init__(
        self,
        data,
        identifier,
        groups=[],
        group_names=None,
        categorical=[],
        continuous=[],
        together=[],
        separate=[],
        enforce_group={},
        minimum_values={},
        maximum_values={},
    ):
        if identifier not in data.columns:
            raise ValueError(f"{identifier=} is not a column name in data.")

        if sum(groups) != len(data):
            raise ValueError(f"{sum(groups)=} is not equal to {len(data)=}.")

        if len(group_names) != len(groups):
            raise ValueError(f"{len(group_names)=} is not equal to {len(groups)=}.")

        if not all(col in data.columns for col in categorical):
            raise ValueError(
                "All elements in categorical must be column names in data."
            )

        if not isinstance(continuous, collections.abc.Iterable):
            raise TypeError(f"Unsupported {type(continuous)=}.")
        if not all(i in data.columns for i in continuous):
            raise ValueError("All elements in continuous must be column names in data.")

        if not isinstance(together, collections.abc.Iterable):
            raise TypeError(f"Unsupported {type(together)=}.")
        if not all(data[identifier].to_list() for i in sum(together, start=[])):
            raise ValueError(
                "All elements in flattened together must be in data[identifier]."
            )

        if not isinstance(separate, collections.abc.Iterable):
            raise TypeError(f"Unsupported {type(separate)=}.")

        if not all(k in data[identifier].to_list() for k in enforce_group):
            raise ValueError("Keys must be valid identifiers.")

        self.data = data
        self.identifier = identifier
        self.groups = groups
        self.group_names = group_names
        self.categorical = categorical
        self.continuous = continuous
        self.together = together
        self.separate = separate
        self.enforce_group = enforce_group
        self.minimum_values = minimum_values
        self.maximum_values = maximum_values

        # Do the initial split.
        self.data["group_number"] = random.sample(
            sum((n * [i] for i, n in enumerate(groups)), start=[]), sum(groups)
        )

        # Name the groups.
        if group_names is not None:
            for i, name in enumerate(group_names):
                self.data.loc[self.data["group_number"] == i, "group_name"] = name

    @property
    def n_groups(self):
        """Number of groups.

        Returns
        -------
        int
            The number of groups.

        """
        return len(self.groups)

    def __getitem__(self, group):
        """DataFrame consisting of all group members.

        Parameters
        ----------
        group: int, str
            Group number or name.

        Returns
        -------
        pd.DataFrame
            Group

        """
        group = self.group_names.index(group) if isinstance(group, str) else group

        return self.data[self.data["group_number"] == group]

    def members(self, group, /):
        """Identifiers of groups members.

        Parameters
        ----------
        group: int, str
            Group number or name.

        Returns
        -------
        list
            Members

        """
        return self[group][self.identifier].to_list()

    @functools.cached_property
    def diversity(self):
        return dict(
            **{i: self.data[i].sum() / len(self.data) for i in self.categorical},
            **{f"{i}_mean": self.data[i].mean() for i in self.continuous},
            **{f"{i}_std": self.data[i].std() for i in self.continuous},
        )

    def group_cost(self, group, /):
        """Cost value of a single group.

        Parameters
        ----------
        group: int, str
            Group number or name.

        Returns
        -------
        float
            Cost

        """
        # Keeping people together or separate.
        c = sum(len(set(i) & set(self.members(group))) ** 2 for i in self.separate)
        c += sum(-(len(set(i) & set(self.members(group))) ** 2) for i in self.together)

        # Enforcing group membership.
        for k, v in self.enforce_group.items():
            if k in self[group][self.identifier].to_list():
                if v != len(self[group]):
                    c += 1e4 * len(self[group])

        # Diversity in the group.
        grp = dict(
            **{i: self[group][i].sum() / len(self[group]) for i in self.categorical},
            **{f"{i}_mean": self[group][i].mean() for i in self.continuous},
            **{f"{i}_std": self[group][i].std() for i in self.continuous},
        )

        c += sum([abs(grp[i] - self.diversity[i]) for i in self.categorical])

        # Means. We multiply by 1 / self.data[i].mean() to ensure this cost does not
        # overshadow the categorical ones.
        c += sum(
            1
            / self.data[i].mean()
            * abs(grp[f"{i}_mean"] - self.diversity[f"{i}_mean"])
            for i in self.continuous
        )

        # Standard deviations.
        c += sum(
            1 / self.data[i].mean() * abs(grp[f"{i}_std"] - self.diversity[f"{i}_std"])
            for i in self.continuous
        )

        c += sum(
            (0 < self[group][i].sum() < j) * 1e3 for i, j in self.minimum_values.items()
        )
        c += sum(
            (self[group][i].sum() > j) * 1e3 for i, j in self.maximum_values.items()
        )

        return c

    @property
    def cost(self):
        """Total cost across all groups.

        Returns
        -------
        float
            Cost

        """
        return sum(self.group_cost(i) for i in range(self.n_groups))

    def step(self):
        """Relaxation step.

        Swap two random people between two different groups. If the cost
        function decreases, the step is accepted.

        """
        n = len(self.data)  # total number of people

        # Select two people from different groups.
        while True:
            a = random.randint(0, n - 1)
            b = random.randint(0, n - 1)

            # groups to which a and b belong to
            ga = self.data.iloc[a]["group_number"]
            gb = self.data.iloc[b]["group_number"]

            if ga != gb:
                break

        # cost before the swap
        c_init = self.group_cost(ga) + self.group_cost(gb)

        # Swap.
        self.data.at[a, "group_number"], self.data.at[b, "group_number"] = gb, ga
        self.data.at[a, "group_name"], self.data.at[b, "group_name"] = (
            self.group_names[gb],
            self.group_names[ga],
        )

        # Go back to original groups if cost does not go down.
        if self.group_cost(ga) + self.group_cost(gb) > c_init:
            self.data.at[a, "group_number"], self.data.at[b, "group_number"] = ga, gb
            self.data.at[a, "group_name"], self.data.at[b, "group_name"] = (
                self.group_names[ga],
                self.group_names[gb],
            )

    def solve(self, n_iter, n_print=500):
        """Build teams.

        Parameters
        ----------
        n_iter: int
            Total number of steps.
        n_print: int
            The cost is printed every n_print number of steps.

        """
        for i in range(n_iter):
            self.step()

            if i % n_print == 0 or i == n_iter - 1:
                print(f"step = {i: 6d}: cost = {self.cost: .3f}")

    def overview(self):
        """Groups overview.

        Returns
        -------
        pd.DataFrame
            Groups overview.

        """
        per_group = dict(
            **{i: self.data.groupby("group_name")[i].sum() for i in self.categorical},
            **{
                f"{i}_mean": self.data.groupby("group_name")[i].mean()
                for i in self.continuous
            },
            **{
                f"{i}_std": self.data.groupby("group_name")[i].std()
                for i in self.continuous
            },
            n=self.data.groupby("group_name").count()[self.categorical[0]],
        )

        totals = dict(
            **{i: self.data[i].count() for i in self.categorical},
            **{f"{i}_mean": self.data[i].mean() for i in self.continuous},
            **{f"{i}_std": self.data[i].std() for i in self.continuous},
            n=len(self.data),
        )

        # Explore how many separate rules are broken.
        split_stats = collections.defaultdict(list)
        for group in self.data["group_name"].unique():
            split_stats["group_name"].append(group)
            split_stats["n_together"].append(
                max(
                    len(set(separate).intersection(set(self.members(group))))
                    for separate in self.separate
                )
            )

        res = pd.concat(
            [
                pd.DataFrame(per_group).join(
                    pd.DataFrame(split_stats).set_index("group_name")
                ),
                pd.DataFrame(totals, index=["total"]),
            ]
        ).fillna(0)
        res.loc["total", "n_together"] = res["n_together"].max()

        return res
