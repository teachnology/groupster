from tqdm import tqdm


class Solver:
    def __init__(self):
        pass

    def __repr__(self):
        pass

    def _step(self, cohort, cost_fn=None):
        # Select two people at random from different groups.
        pair = cohort.data.sample(n=2, replace=False)

        a = pair.iloc[0]
        b = pair.iloc[1]

        # Cost before the swap.
        cohort_diversity = cohort.diversity()
        cost_before = cohort[a.group].cost(
            cost_fn=cost_fn, cohort_diversity=cohort_diversity
        ) + cohort[b.group].cost(cost_fn=cost_fn, cohort_diversity=cohort_diversity)

        # Swap the groups of two people.
        cohort.data.loc[a.name, "group"] = b.group
        cohort.data.loc[b.name, "group"] = a.group

        # Cost after the swap.
        cost_after = cohort[a.group].cost(
            cost_fn=cost_fn, cohort_diversity=cohort_diversity
        ) + cohort[b.group].cost(cost_fn=cost_fn, cohort_diversity=cohort_diversity)

        if cost_after < cost_before:
            # If the cost has decreased, keep the swap.
            return True
        else:
            # If the cost has increased, revert the swap.
            cohort.data.loc[a.name, "group"] = a.group
            cohort.data.loc[b.name, "group"] = b.group
            return False

        # Use the previous cost for the next step to speed things up.

    def solve(self, cohort, n, cost_fn=None):
        progress_bar = tqdm(range(n), desc="Solving")
        progress_bar.set_postfix(
            {
                "Cost": "pending",
                "Acceptance rate": "pending",
            }
        )

        accepted = 0
        for i in progress_bar:
            accepted += self._step(cohort, cost_fn=cost_fn)

            if i % (n // 10) == 0:
                if i > 0:
                    cost = cohort.cost(cost_fn=cost_fn)
                    acceptance_rate = accepted / (n // 10)
                else:
                    cost = "pending"
                    acceptance_rate = "pending"

                progress_bar.set_postfix(
                    {
                        "Cost": cost,
                        "Acceptance rate": acceptance_rate,
                    }
                )
                accepted = 0
