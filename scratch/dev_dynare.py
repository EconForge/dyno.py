import dyno
from dyno.cli import dynare

filename = dyno.examples_path("modfiles", "example1.mod")

dynare(filename, strict=False)



report = dynare(filename)
print(report.eigenvalues)