from dyno.report import dsge_report
from dyno import examples_path

print("=== DSGE Report for example2.mod (dynare preprocessor)===")
report = dsge_report(
    filename=examples_path("modfiles", "example2.mod"),
    modfile_preprocessor="dynare",
)

print("=== DSGE Report for example2.mod (Dyno Model)===")
report = dsge_report(
    filename=examples_path("modfiles", "example2.mod"),
    modfile_preprocessor="dyno",
)

print(report)
