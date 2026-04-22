from dyno import DynoModel

txt = """
a <- 0.1

[type=transition] {

    x[t] = x[t-1] + e[t]  [name=Productivity]

}

y[t] = y[t-1] + e[t]  
z[t] = z[t-1] + e[t]  [Production]




e[t] <- N(0,1)

@name: Demo
"""
model = DynoModel(txt=txt)
model.print_equations_with_tags()

