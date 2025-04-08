import pathlib

import fakeitmakeit as fm

CWD = pathlib.Path(__file__).parent

if __name__ == "__main__":
    if not pathlib.Path(CWD / "students.csv").exists():
        cohort = fm.cohort(n=49)
        marks = fm.assignment(usernames=cohort.index)

        cohort = (
            cohort.assign(
                female=lambda df_: df_.gender.eq("female"),
                edsml=lambda df_: df_.course.eq("edsml"),
                name=lambda df_: df_.first_name + " " + df_.last_name,
            )
            .merge(marks, left_index=True, right_index=True)
            .loc[:, ["name", "female", "edsml", "mark"]]
        )

        cohort.to_csv(CWD / "cohort.csv")
