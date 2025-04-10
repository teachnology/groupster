from tqdm import tqdm


class Solver:
    """A solver class to minimise the cost of a cohort.

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
    diversity_cost_fn : callable, optional
        A function to compute the diversity cost of a group. The default is
        `util.diversity_cost`.
    restriction_cost_fn : callable, optional
        A function to compute the restriction cost of a group. The default is
        `util.restriction_cost`.

    """

    def __init__(
        self,
        keep_together=None,
        keep_separate=None,
        bool_min=None,
        diversity_cost_fn=None,
        restriction_cost_fn=None,
    ):
        self.keep_together = keep_together
        self.keep_separate = keep_separate
        self.bool_min = bool_min

        self.diversity_cost_fn = diversity_cost_fn
        self.restriction_cost_fn = restriction_cost_fn

        self._cached_cost = {}

    def _cost(self, group, cohort, use_cache=False):
        """Compute the cost of a group.

        The cost is computed as the sum of the diversity cost and the restriction cost.
        The cost is cached to avoid recomputing it.

        Parameters
        ----------
        group : Group
            The group to compute the cost for.
        cohort : Cohort
            The cohort used to compute the diversity cost.
        use_cache : bool, optional
            Whether to use the cached cost. The default is False.

        Returns
        -------
        float
            The cost of the group.
        """
        group_name = group.data.group.iloc[0]

        if not use_cache:
            cost = group.diversity_cost(
                cohort_diversity=cohort.diversity, cost_fn=self.diversity_cost_fn
            ) + group.restriction_cost(
                keep_together=self.keep_together,
                keep_separate=self.keep_separate,
                bool_min=self.bool_min,
                cost_fn=self.restriction_cost_fn,
            )

            self._cached_cost[group_name] = cost

            return cost

        if group_name not in self._cached_cost:
            return self._cost(group, cohort, use_cache=False)

        else:
            return self._cached_cost[group_name]

    def _step(self, cohort):
        """Perform a single step of the algorithm.

        This method selects two people at random from different groups and swaps their
        groups. If the cost of the new groups is lower than the cost of the old groups,
        the swap is accepted. Otherwise, the swap is rejected.

        Parameters
        ----------
        cohort : Cohort
            The cohort to solve.

        Returns
        -------
        bool
            Whether the swap was accepted or not.
        """
        # Select two people at random from different groups.
        while True:
            [(_, a), (_, b)] = cohort.data.sample(n=2, replace=False).iterrows()
            if a.group != b.group:
                break

        # Cost before the swap.
        cost_before = self._cost(cohort[a.group], cohort, use_cache=False) + self._cost(
            cohort[b.group], cohort, use_cache=False
        )

        # Swap the groups of two people.
        cohort.data.loc[a.name, "group"] = b.group
        cohort.data.loc[b.name, "group"] = a.group

        # Cost after the swap.
        cost_after = self._cost(cohort[a.group], cohort, use_cache=False) + self._cost(
            cohort[b.group], cohort, use_cache=False
        )
        if cost_after < cost_before:
            # If the cost has decreased, keep the swap.
            return True
        else:
            # If the cost has increased, revert the swap.
            cohort.data.loc[a.name, "group"] = a.group
            cohort.data.loc[b.name, "group"] = b.group
            return False

    def solve(self, cohort, n):
        """Solve the cohort by minimising the cost.

        This method performs a number of steps to minimise the cost of the cohort. The
        number of steps is specified by the ``n`` parameter. Each step is performed by
        ``_step`` method.

        The Progress bar updates the cost and acceptance rate every 10% of the steps.

        Parameters
        ----------
        cohort : Cohort
            The cohort to solve.
        n : int
            The number of steps to perform.

        """
        progress_bar = tqdm(range(n), desc="Solving")
        progress_bar.set_postfix(
            {
                "Cost": "pending",
                "Acceptance rate": "pending",
            }
        )

        accepted = 0
        for i in progress_bar:
            accepted += self._step(cohort)

            if i % (n // 10) == 0:
                if i > 0:
                    diversity_cost = cohort.diversity_cost(
                        cost_fn=self.diversity_cost_fn
                    )
                    restriction_cost = cohort.restriction_cost(
                        keep_together=self.keep_together,
                        keep_separate=self.keep_separate,
                        bool_min=self.bool_min,
                        cost_fn=self.restriction_cost_fn,
                    )
                    acceptance_rate = accepted / (n // 10)
                else:
                    diversity_cost = "pending"
                    restriction_cost = "pending"
                    acceptance_rate = "pending"

                progress_bar.set_postfix(
                    {
                        "diversity_cost": diversity_cost,
                        "restriction_cost": restriction_cost,
                        "acceptance_rate": acceptance_rate,
                    }
                )
                accepted = 0
