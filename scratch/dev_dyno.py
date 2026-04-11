from dyno import DynoModel

import time

start = time.time()
report = DynoModel("examples/neo.dyno").run()
end = time.time()

print(f"Execution time: {end - start} seconds")

