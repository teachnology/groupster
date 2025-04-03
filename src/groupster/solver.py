from tqdm import tqdm


class Solver:
    def __init__(self, keep_together=None, keep_separate=None):
        self.keep_together = list(keep_together) if keep_together is not None else None
        self.keep_separate = list(keep_separate) if keep_separate is not None else None

    def _step(self, cohort, diversity_cost_fn=None, restriction_cost_fn=None):
        # Select two people at random from different groups.
        while True:
            pair = cohort.data.sample(n=2, replace=False)
            a = pair.iloc[0]
            b = pair.iloc[1]
            if a.group != b.group:
                break

        # Cost before the swap.
        cohort_diversity = cohort.diversity()
        diversity_cost_before = cohort[a.group].diversity_cost(
            cost_fn=diversity_cost_fn, cohort_diversity=cohort_diversity
        ) + cohort[b.group].diversity_cost(
            cost_fn=diversity_cost_fn, cohort_diversity=cohort_diversity
        )

        restriction_cost_before = cohort[a.group].restriction_cost(
            keep_together=self.keep_together,
            keep_separate=self.keep_separate,
            cost_fn=restriction_cost_fn,
        ) + cohort[b.group].restriction_cost(
            keep_together=self.keep_together,
            keep_separate=self.keep_separate,
            cost_fn=restriction_cost_fn,
        )

        # Swap the groups of two people.
        cohort.data.loc[a.name, "group"] = b.group
        cohort.data.loc[b.name, "group"] = a.group

        # Cost after the swap.
        diversity_cost_after = cohort[a.group].diversity_cost(
            cost_fn=diversity_cost_fn, cohort_diversity=cohort_diversity
        ) + cohort[b.group].diversity_cost(
            cost_fn=diversity_cost_fn, cohort_diversity=cohort_diversity
        )

        restriction_cost_after = cohort[a.group].restriction_cost(
            keep_together=self.keep_together,
            keep_separate=self.keep_separate,
            cost_fn=restriction_cost_fn,
        ) + cohort[b.group].restriction_cost(
            keep_together=self.keep_together,
            keep_separate=self.keep_separate,
            cost_fn=restriction_cost_fn,
        )

        if (diversity_cost_after + restriction_cost_after) < (
            diversity_cost_before + restriction_cost_before
        ):
            # If the cost has decreased, keep the swap.
            return True
        else:
            # If the cost has increased, revert the swap.
            cohort.data.loc[a.name, "group"] = a.group
            cohort.data.loc[b.name, "group"] = b.group
            return False

        # Use the previous cost for the next step to speed things up.

    def solve(self, cohort, n, diversity_cost_fn=None, restriction_cost_fn=None):
        progress_bar = tqdm(range(n), desc="Solving")
        progress_bar.set_postfix(
            {
                "Cost": "pending",
                "Acceptance rate": "pending",
            }
        )

        accepted = 0
        for i in progress_bar:
            accepted += self._step(
                cohort,
                diversity_cost_fn=diversity_cost_fn,
                restriction_cost_fn=restriction_cost_fn,
            )

            if i % (n // 10) == 0:
                if i > 0:
                    diversity_cost = cohort.diversity_cost(cost_fn=diversity_cost_fn)
                    restriction_cost = cohort.restriction_cost(
                        keep_together=self.keep_together,
                        keep_separate=self.keep_separate,
                        cost_fn=restriction_cost_fn,
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
