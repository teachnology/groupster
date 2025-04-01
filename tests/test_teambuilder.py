import pathlib

import pandas as pd
import pytest
import teambuilder as tb


@pytest.fixture(scope="package")
def data():
    dirname = pathlib.Path(__file__).parent / "data"
    cohort = pd.read_csv(dirname / "cohort.csv")

    # Keep only EDSML and GEMS.
    sh = cohort[cohort.course.isin(["edsml", "gems"])].reset_index(drop=True)

    # Encode categorical variables.
    sh["female"] = sh.gender == "female"
    sh["gems"] = sh.course == "gems"

    # Return only relevant columns.
    return sh[
        [
            "username",
            "name",
            "female",
            "gems",
            "mark",
        ]
    ]


class TestTeamBuilder:
    def test_data(self, data):
        print(f"{len(data) = }")  # 147
        print(f"{len(data[data.gems]) = }")  # 47
        print(f"{len(data[~data.gems]) = }")  # 100
        print(f"{len(data[data.female]) = }")  # 52
        print(f"{len(data[~data.female]) = }")  # 95
        print(f"{data.mark.mean() = }")  # 64.70
        print(f"{data.mark.std() = }")  # 7.59
        assert True

    def test_solve_full(self, data):
        groups = 12 * [10] + 3 * [9]
        group_names = [f"group_{i}" for i in range(len(groups))]

        solver = tb.TeamBuilder(
            data,
            identifier="username",
            groups=groups,
            group_names=group_names,
            categorical=["gems", "female"],
            continuous=["mark"],
            together=[],
            separate=[],
            enforce_group={},
        )

        solver.solve(n_iter=2_000)

        # Do checks.
        res = solver.data

        # Check that the groups are the right size.
        assert all(res.groupby("group_number").size() == groups)

        # Check that categorical variables are equally distributed.
        for group in group_names:
            assert solver[group].gems.sum() in (3, 4)
            assert solver[group].female.sum() in (3, 4)

        # Check that continuous variable mean and stdev is respected.
        for group in group_names:
            assert abs(solver[group].mark.mean() - 64.70) < 5
            assert abs(solver[group].mark.std() - 7.59) < 4
