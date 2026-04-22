from dyno import DynoModel

txt = """


a <- 0.1 :: "This is a parameter"
b <- 0.1 :: [label="This is a parameter", in=R+]


x[t] = x[t-1] + e[t]  :: "Productivity"

y[t] = y[t-1] + e[t]  :: Production
z[t] = z[t-1] + e[t]  :: Inventory

# e[t] <- N(0,1)

e[0] <- 0.0
e[1] <- 0.1
e[2] <- 0.0


@name: Demo
"""
model = DynoModel(txt=txt)
model.print_equations_with_tags()

