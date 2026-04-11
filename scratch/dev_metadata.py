from dyno import DynoModel, DynareModel
import json


# Test 1: DynareModel with undeclared params
model = DynareModel("examples/modfiles/example1.mod", allow_undeclared_params=True)
print("=== DynareModel example1.mod (with allow_undeclred_params) ===")
print(json.dumps(model.metadata, indent=2))
print()

# Test 2: DynareModel with options parsing
model2 = DynareModel("examples/modfiles/example2.mod", allow_undeclared_params=True)
print("=== DynareModel example2.mod (with options) ===")
print(json.dumps(model2.metadata, indent=2))
print()

# Test 3: DynareModel without option
model3 = DynareModel("examples/modfiles/example2.mod")
print("=== DynareModel commands ===")
print(f"Commands: {model3.metadata.get('dynare_commands', [])}")
print()

# Test 4: Access specific commands
print("=== Accessing specific commands ===")
for cmd in model2.metadata['dynare_commands']:
    print(f"Command: {cmd['command']}, Options: {cmd['options']}")




report = DynareModel("examples/modfiles/example2.mod").run()


from dyno import DynoModel

DynoModel("examples/neo.dyno").run()




from dyno.report import dsge_report

dsge_report(
    filename="examples/modfiles/example2.mod"
)


# model2.residuals

# model2.check()

# model2_copy = model2.steady()


# model2.residuals



# model2.steady().check().solve()
# ms = model2.symbolic



# report = model2.run()


