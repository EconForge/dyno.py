from dyno.yamlfile import YAMLFile

model = YAMLFile('examples/neo.yaml')

print(model)

dr = model.solve()

print(dr)