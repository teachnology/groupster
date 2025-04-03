from tqdm import tqdm


class Solver:
    def __init__(
        self,
        keep_together=None,
        keep_separate=None,
        diversity_cost_fn=None,
        restriction_cost_fn=None,
    ):
        self.keep_together = keep_together
        self.keep_separate = keep_separate
        self.diversity_cost_fn = diversity_cost_fn
        self.restriction_cost_fn = restriction_cost_fn

        self._cached_cost = {}

    def _cost(self, group, cohort, use_cache=False):
        group_name = group.data.group.iloc[0]

        if not use_cache:
            cost = group.diversity_cost(
                cohort_diversity=cohort.diversity, cost_fn=self.diversity_cost_fn
            ) + group.restriction_cost(
                keep_together=self.keep_together,
                keep_separate=self.keep_separate,
                cost_fn=self.restriction_cost_fn,
            )

            self._cached_cost[group_name] = cost

            return cost

        if group_name not in self._cached_cost:
            return self._cost(group, cohort, use_cache=False)

        else:
            return self._cached_cost[group_name]

    def _step(self, cohort):
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
