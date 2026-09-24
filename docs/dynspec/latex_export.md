# LaTeX Rendering & Export

DynSpec provides automatic conversion of parsed AST equation trees into publication-ready LaTeX markup using `dyno.dynspec.latex`.

---

## The `LatexTransformer`

The `LatexTransformer` is a Lark visitor that walks expression and equation trees to construct canonical LaTeX formulas with correct mathematical typography.

```python
from dyno.dynspec.latex import latex

# Convert an equation string directly
tex_code = latex("1/c[t] = beta * (1/c[t+1]) * (alpha*y[t+1]/k[t] + 1 - delta)")
print(tex_code)
```

**Output:**
```latex
\frac{1}{c_{t}} = \beta \frac{1}{c_{t+1}} \left(\frac{\alpha y_{t+1}}{k_{t}} + 1 - \delta\right)
```

Rendered result:

$$\frac{1}{c_{t}} = \beta \frac{1}{c_{t+1}} \left(\frac{\alpha y_{t+1}}{k_{t}} + 1 - \delta\right)$$

---

## Typography & Formatting Features

### 1. Automatic Greek Character Mapping
ASCII symbol names matching Greek letter names are automatically converted to their LaTeX equivalents:
- `alpha` $\implies$ `\alpha` ($\alpha$)
- `beta` $\implies$ `\beta` ($\beta$)
- `gamma` $\implies$ `\gamma` ($\gamma$)
- `delta` $\implies$ `\delta` ($\delta$)
- `rho` $\implies$ `\rho` ($\rho$)
- `epsilon` $\implies$ `\epsilon` ($\epsilon$)

### 2. Time Subscript Formatting
Time indices and shifts are formatted as clean LaTeX subscripts:
- `k[t]` $\implies$ `k_{t}` ($k_t$)
- `k[t-1]` $\implies$ `k_{t-1}` ($k_{t-1}$)
- `c[t+1]` $\implies$ `c_{t+1}` ($c_{t+1}$)
- `k[~]` $\implies$ `\bar{k}` ($\bar{k}$)
- `k[0]` $\implies$ `k_{0}` ($k_0$)

### 3. Smart Fractions & Parentheses
- Divisions are transformed into `\frac{numerator}{denominator}`.
- Operator precedence rules automatically suppress redundant nested parentheses while preserving necessary algebraic groupings using `\left(` and `\right)`.

---

## Programmatic Usage

Render full equation sets from a parsed model:

```python
from dyno import DynoModel
from dyno.dynspec.latex import latex

model = DynoModel("examples/neo.dyno")

# Print all equations as a LaTeX align block
print("\\begin{align}")
for eq_tree in model.symbolic.equations:
    print(f"  {latex(eq_tree)} \\\\")
print("\\end{align}")
```
