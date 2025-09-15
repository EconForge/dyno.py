# from dyno.yamlfile import YAMLFile

# model = YAMLFile('examples/neo.yaml')

# print(model)

# dr = model.solve()

# print(dr)

from dyno.report import dsge_report

report = dsge_report(filename='RBC.dyno')
print(report)

print(report._repr_html_())

report = dsge_report(filename='RBC.dyno')
print(report)

print(report._repr_html_())