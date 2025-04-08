import pathlib

import numpy as np
import pandas as pd

import groupster as gr

CWD = pathlib.Path(__file__).parent
data = pd.read_csv(CWD / "data" / "cohort.csv", index_col="username")


class TestInit:
    def test_group(self):
        group = gr.Group(data=data.iloc[:10, :], bools=["female"], nums=["mark"])
        assert isinstance(group, gr.Group)

        assert group.bools == ["female"]
        assert group.nums == ["mark"]


class TestDiversity:
    def test_diversity(self):
        group = gr.Group(data=data.iloc[:10, :], bools=["female"], nums=["mark"])

        assert isinstance(group.diversity, pd.Series)
        assert len(group.diversity) == 3

        assert group.diversity.loc["bool_female"] == 0.2
        assert np.isclose(group.diversity.loc["num_mark_mean"], 64.381)
        assert np.isclose(group.diversity.loc["num_mark_std"], 4.2121)


class TestCosts:
    def test_diversity_cost(self):
        cohort = gr.Cohort(
            data=data, groups={"g1": len(data)}, bools=["female"], nums=["mark"]
        )

        group = gr.Group(data=data.iloc[:10, :], bools=["female"], nums=["mark"])

        cost = (
            abs(0.2653 - 0.2)  # female
            + abs(64.381 - 65.7451) / 65.7451  # mark mean
            + abs(4.2121 - 5.38341) / 65.7451  # mark std
        )
        assert np.isclose(
            group.diversity_cost(cohort_diversity=cohort.diversity),
            cost,
            atol=0.01,
            rtol=0.0,
        )

    def test_restriction_cost(self):
        keep_together = [
            ["aek32", "juf35"],  # satisfied
            ["jxx1444", "yz9097"],  # not satisfied
        ]
        keep_separate = [
            ["aek32", "juf35"],  # not satisfied
            ["jxx1444", "yz9097"],  # satisfied
        ]
        bool_min = {"female": 3}  # not satisfied

        group = gr.Group(data=data.iloc[:10, :], bools=["female"], nums=["mark"])

        cost = (
            -(2**2)
            - 1**2  # keep_separate
            + 1**2
            + 2**2  # keep_together
            + 1 * 10  # bool_min
        )
        assert np.isclose(
            group.restriction_cost(
                keep_together=keep_together,
                keep_separate=keep_separate,
                bool_min=bool_min,
            ),
            cost,
        )
