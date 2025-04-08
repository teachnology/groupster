import pathlib

import numpy as np
import pandas as pd

import groupster as gr

CWD = pathlib.Path(__file__).parent
data = pd.read_csv(CWD / "data" / "cohort.csv", index_col="username")
groups = {"g1": 10, "g2": 10, "g3": 10, "g4": 10, "g5": 9}


class TestInit:
    def test_group(self):
        cohort = gr.Cohort(data=data, groups=groups)
        assert isinstance(cohort, gr.Cohort)
        assert cohort.data.drop(columns=["group"]).equals(data)

        assert "group" in cohort.data.columns
        assert cohort.data.group.nunique() == 5
        assert cohort.data.group.value_counts().to_dict() == groups


class TestGroup:
    def test_getitem(self):
        cohort = gr.Cohort(data=data, groups=groups)

        group = cohort["g1"]
        assert isinstance(group, gr.Group)
        assert group.data.shape[0] == 10
        assert group.data.group.eq("g1").all()

        assert cohort["g5"].data.index.size == 9


class TestDiversity:
    def test_bools(self):
        cohort = gr.Cohort(data=data, groups=groups, bools=["female"])

        assert isinstance(cohort.diversity, pd.Series)
        assert len(cohort.diversity) == 1
        assert cohort.diversity.loc["bool_female"] * len(data) == 13.0

    def test_nums(self):
        cohort = gr.Cohort(data=data, groups=groups, nums=["mark"])

        assert isinstance(cohort.diversity, pd.Series)
        assert len(cohort.diversity) == 2
        assert np.isclose(cohort.diversity.loc["num_mark_mean"], 65.7451)
        assert np.isclose(cohort.diversity.loc["num_mark_std"], 5.38340)


class TestOverview:
    def test_overview(self):
        cohort = gr.Cohort(
            data, groups=groups, bools=["female", "edsml"], nums=["mark"]
        )
        overview = cohort.overview()
        assert isinstance(overview, pd.DataFrame)
        assert overview.shape == (5, 6)
        assert overview.index.name == "group"
