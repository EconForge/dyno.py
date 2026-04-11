from dyno import DynoModel, examples_path

import time

start = time.time()
report = DynoModel(examples_path("neo.dyno")).run()
end = time.time()

print(f"Execution time: {end - start} seconds")

start = time.time()

report = DynoModel(examples_path("neo.dyno.yaml")).run()
end = time.time()

print(f"Execution time: {end - start} seconds")
