import pathlib

import groupster as gr
import numpy as np
import pandas as pd

CWD = pathlib.Path(__file__).parent
data = pd.read_csv(CWD / "data" / "cohort.csv", index_col="username")
groups = {"g1": 10, "g2": 10, "g3": 10, "g4": 10, "g5": 9}


class TestBools:
    def test_solve(self):
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


class TestNums:
    def test_solve(self):
        cohort = gr.Cohort(data=data, groups=groups, nums=["mark"])
        solver = gr.Solver()

        mean_initial_std = (
            cohort.data.groupby("group").mark.mean().sub(cohort.data.mark.mean()).std()
        )
        std_initial_std = (
            cohort.data.groupby("group").mark.std().sub(cohort.data.mark.std()).std()
        )

        solver.solve(cohort=cohort, n=500)

        mean_final_std = (
            cohort.data.groupby("group").mark.mean().sub(cohort.data.mark.mean()).std()
        )
        std_final_std = (
            cohort.data.groupby("group").mark.std().sub(cohort.data.mark.std()).std()
        )

        # We expect the ranges to decrease.
        assert mean_final_std < mean_initial_std
        assert std_final_std < std_initial_std


class TestMixed:
    def test_solve(self):
        cohort = gr.Cohort(data=data, groups=groups, bools=["female"], nums=["mark"])
        solver = gr.Solver()

        mean_initial_std = (
            cohort.data.groupby("group").mark.mean().sub(cohort.data.mark.mean()).std()
        )
        std_initial_std = (
            cohort.data.groupby("group").mark.std().sub(cohort.data.mark.std()).std()
        )

        solver.solve(cohort=cohort, n=500)

        mean_final_std = (
            cohort.data.groupby("group").mark.mean().sub(cohort.data.mark.mean()).std()
        )
        std_final_std = (
            cohort.data.groupby("group").mark.std().sub(cohort.data.mark.std()).std()
        )

        # We expect the ranges to decrease.
        assert mean_final_std < mean_initial_std
        assert std_final_std < std_initial_std

        assert cohort.data.groupby("group").female.sum().between(2, 3).all()
