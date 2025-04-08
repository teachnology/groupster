import pathlib

import pandas as pd
import pytest

import groupster as gr

CWD = pathlib.Path(__file__).parent


@pytest.fixture(scope="function")
def data():
    return pd.read_csv(CWD / "data" / "cohort.csv", index_col="username")


@pytest.fixture(scope="function")
def groups():
    return {"g1": 10, "g2": 10, "g3": 10, "g4": 10, "g5": 9}


class TestDiversity:
    def test_bools(self, data, groups):
        cohort = gr.Cohort(data=data, groups=groups, bools=["female"])
        solver = gr.Solver()

        initial_df = cohort.data.copy()

        solver.solve(cohort=cohort, n=500)

        # Ensure data is not corrupted.
        assert cohort.data.drop(columns=["group"]).equals(
            initial_df.drop(columns=["group"])
        )

        # Ensure that the group sizes are the same.
        assert cohort.data.groupby("group").size().to_dict() == groups

        # Total number of females.
        assert cohort.data.groupby("group").female.sum().sum() == 13

        # We expect either 2 or 3 females in each group.
        assert cohort.data.groupby("group").female.sum().between(2, 3).all()
        assert cohort.overview().female.between(2, 3).all()

    def test_nums(self, data, groups):
        cohort = gr.Cohort(data=data, groups=groups, nums=["mark"])
        solver = gr.Solver()

        initial_overview = cohort.overview()
        solver.solve(cohort=cohort, n=2000)
        final_overview = cohort.overview()

        # We expect the variability (std) of mean and std to decrease.
        assert final_overview["mean(mark)"].std() < initial_overview["mean(mark)"].std()
        assert final_overview["std(mark)"].std() < initial_overview["std(mark)"].std()

    def test_mixed(self, data, groups):
        cohort = gr.Cohort(data=data, groups=groups, bools=["female"], nums=["mark"])
        solver = gr.Solver()

        initial_overview = cohort.overview()
        solver.solve(cohort=cohort, n=2000)
        final_overview = cohort.overview()

        # We expect the variability (std) of mean and std to decrease.
        assert final_overview["mean(mark)"].std() < initial_overview["mean(mark)"].std()
        assert final_overview["std(mark)"].std() < initial_overview["std(mark)"].std()

        assert final_overview.female.between(2, 3).all()


class TestRestriction:
    def test_keep_together(self, data, groups):
        cohort = gr.Cohort(data=data, groups=groups)
        keep_together = [["ff402", "yjt99", "cr947"], ["jr848", "fs81"]]
        solver = gr.Solver(keep_together=keep_together)

        solver.solve(cohort=cohort, n=2000)

        for subset in keep_together:
            assert cohort.data.loc[subset, "group"].value_counts().size == 1

    def test_keep_separate(self, data, groups):
        cohort = gr.Cohort(data=data, groups=groups)
        keep_separate = [["ff402", "yjt99", "cr947"], ["jr848", "fs81"]]
        solver = gr.Solver(keep_separate=keep_separate)

        solver.solve(cohort=cohort, n=2000)

        for subset in keep_separate:
            assert cohort.data.loc[subset, "group"].value_counts().size == len(subset)

    def test_bool_min(self, data):
        groups = {f"g{i}": 7 for i in range(1, 8)}
        cohort = gr.Cohort(data=data, groups=groups, bools=["edsml"])
        solver = gr.Solver(bool_min={"edsml": 3})

        solver.solve(cohort=cohort, n=2000)

        assert cohort.overview().edsml.value_counts().index.isin([3, 0]).all()


class TestAll:
    def test_solve(self, data, groups):
        cohort = gr.Cohort(
            data=data, groups=groups, bools=["female", "edsml"], nums=["mark"]
        )
        keep_together = [["ff402", "yjt99", "cr947"], ["jr848", "fs81"]]
        keep_separate = [["yz9097", "ay631", "mpc1253"], ["qbk99", "fxg194"]]
        solver = gr.Solver(
            keep_together=keep_together,
            keep_separate=keep_separate,
        )
        initial_overview = cohort.overview()
        solver.solve(cohort=cohort, n=2000)
        final_overview = cohort.overview()

        # We expect the variability (std) of mean and std to decrease.
        assert final_overview["mean(mark)"].std() < initial_overview["mean(mark)"].std()
        assert final_overview["std(mark)"].std() < initial_overview["std(mark)"].std()

        assert final_overview.female.isin([2, 3]).all()
        assert final_overview.edsml.isin([3, 4]).all()

        for subset in keep_together:
            assert cohort.data.loc[subset, "group"].value_counts().size == 1

        for subset in keep_separate:
            assert cohort.data.loc[subset, "group"].value_counts().size == len(subset)
