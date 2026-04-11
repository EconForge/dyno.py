from dyno.report import dsge_report

print("=== DSGE Report for example2.mod (dynare preprocessor)===")
report = dsge_report(
    filename="examples/modfiles/example2.mod",
    modfile_preprocessor="dynare",
    )

print("=== DSGE Report for example2.mod (Dyno Model)===")
report = dsge_report(
    filename="examples/modfiles/example2.mod",
    modfile_preprocessor="dyno",
    )

print(report)