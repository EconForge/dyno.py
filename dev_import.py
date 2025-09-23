from dyno.dynofile2 import LDynoModel

from rich import inspect, print


# import time
# model = LDynoModel("examples/RBC.dyno")
# t1 = time.time()
# model = LDynoModel("examples/RBC.dyno")
# model.solve()
# t2 = time.time()
# print("Elsapsed time:", t2-t1)


# y,e = model.steady_state
# model.compute_derivatives(y,y,y,e)



import time
model = LDynoModel("examples/modfiles/RBC.mod")
t1 = time.time()
model = LDynoModel("examples/modfiles/RBC.mod")
dr = (model.solve())
print(model.symbols)
print(dr.Y)
t2 = time.time()
print("Elsapsed time:", t2-t1)


# # Importing modfile with preprocessor
# from dyno.modfile import DynareModel
# import time
# model = DynareModel("examples/modfiles/RBC.mod")
# t1 = time.time()
# model = DynareModel("examples/modfiles/RBC.mod")
# # print(model.solve())
# # print(model.symbols)
# # print(dr.Y)
# t2 = time.time()
# print("Elsapsed time:", t2-t1)


exit()
