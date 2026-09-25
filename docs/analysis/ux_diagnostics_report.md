# Dyno UX Audit Report: Misspecified & Incomplete Models Catalog

Total catalog cases: **31**

## Summary by Case

| ID | Category | Title | Stage | Raised Error | Warnings | Ideal Diagnostic |
|---|---|---|---|---|---|---|
| **SYN-001** | `syntax` | Top-level equality used instead of assignment operator | `import` | _None_ | 2 warning(s) | Top-level equality detected for 'alpha = 0.35'. In Dyno, '='... |
| **SYN-002** | `syntax` | Parentheses used for time indexing instead of brackets | `import` | `ValueError` | 1 warning(s) | Syntax error: 'k(t)' looks like a time index or lag. Time in... |
| **SYN-003** | `syntax` | Trailing semicolon from MATLAB/Dynare convention | `import` | `LARKParserError` | _None_ | Semicolons are not required or permitted as statement termin... |
| **SYN-004** | `syntax` | Dynare variable declaration block in .dyno file | `import` | `LARKParserError` | _None_ | Dyno does not require variable declarations. Variables are i... |
| **SYN-005** | `syntax` | Dynare implicit lag notation k(-1) | `import` | `ValueError` | _None_ | Dynare notation 'k(-1)' is not recognized in .dyno files. Us... |
| **SYN-006** | `syntax` | Double equals comparison operator in equation | `import` | `LARKParserError` | _None_ | Dynamic equations in Dyno use a single '=' operator (e.g. 'l... |
| **SYN-007** | `syntax` | Reversed assignment arrow -> | `import` | `LARKParserError` | _None_ | Assignment in Dyno must be leftward: 'alpha <- 0.35' or 'alp... |
| **SYN-008** | `syntax` | Unsupported mathematical function | `import` | `ValueError` | _None_ | Function 'sigmoid' is not recognized. Supported functions in... |
| **SYM-001** | `symbol_resolution` | Typo in parameter name in equation definition | `import` | _None_ | 2 warning(s) | Parameter 'alhpa' used in equation is never defined. Did you... |
| **SYM-002** | `symbol_resolution` | Variable in equation without time index | `import` | _None_ | 2 warning(s) | Symbol 'c' appears in equation without a time index [t], but... |
| **SYM-003** | `symbol_resolution` | Out-of-order parameter calibration | `import` | _None_ | 2 warning(s) | Parameter 'b' was referenced on line 2 before its assignment... |
| **SYM-004** | `symbol_resolution` | Circular parameter calibration | `import` | _None_ | 2 warning(s) | Circular dependency detected in parameter calibration betwee... |
| **SYM-005** | `symbol_resolution` | Parameter self-reference without initial calibration | `import` | _None_ | 2 warning(s) | Parameter 'a' references itself before having an established... |
| **SYM-006** | `symbol_resolution` | Typo in variable name creating phantom variable | `check` | `UndefinedSymbolError` | _None_ | Variable 'k_cap[t]' appears in equations but has no steady s... |
| **SS-001** | `steady_state` | Missing steady-state declaration for endogenous variable | `check` | `UndefinedSymbolError` | _None_ | Variable 'k' has no steady-state declaration. Provide 'k[~] ... |
| **SS-002** | `steady_state` | Typo in steady-state declaration name k_ss | `check` | `UndefinedSymbolError` | _None_ | Unused parameter 'k_ss' detected. Did you mean to declare th... |
| **SS-003** | `steady_state` | Inconsistent steady-state value | `check` | `SteadyStateError` | _None_ | Equation 1 has non-zero steady-state residual (residual = -6... |
| **SS-004** | `steady_state` | Steady-state declaration causes division by zero | `check` | `ZeroDivisionError` | _None_ | Division by zero encountered in equation 1 when evaluated at... |
| **SS-005** | `steady_state` | Steady-state evaluation causes negative log domain error | `check` | `ValueError` | _None_ | Domain error: log(z[t]) evaluated with non-positive steady s... |
| **SS-006** | `steady_state` | Forward reference across steady-state declarations | `check` | `UndefinedSymbolError` | _None_ | Steady state 'y[~]' was referenced in 'k[~]' before 'y[~]' w... |
| **SHK-001** | `shocks` | Shock variable in equations without distribution declaration | `solve` | `UndefinedSymbolError` | _None_ | Variable 'e' appears to be an exogenous shock but has no dis... |
| **SHK-002** | `shocks` | Shock distribution too many arguments | `import` | `TypeError` | _None_ | N(...) requires 1 or 2 arguments: N(std) or N(mean, std). E.... |
| **SHK-003** | `shocks` | Shock assigned as parameter without [t] | `solve` | `UndefinedSymbolError` | _None_ | Shock process assigned to constant 'e' rather than time seri... |
| **SHK-004** | `shocks` | Shock distribution function name typo Normal() | `import` | `ValueError` | _None_ | Distribution 'Normal' is not recognized. Use 'N(std)' or 'N(... |
| **SYS-001** | `system_structure` | Underdetermined model (fewer equations than endogenous variables) | `solve` | `SteadyStateError` | _None_ | Model is underdetermined: 1 equation provided for 2 endogeno... |
| **SYS-002** | `system_structure` | Overdetermined model (more equations than endogenous variables) | `solve` | `ValueError` | _None_ | Model is overdetermined: 2 equations provided for 1 endogeno... |
| **SYS-003** | `system_structure` | Higher-order lag k[t-2] without auxiliary state | `import` | _None_ | 1 warning(s) | Higher-order lag 'k[t-2]' detected. Dyno first-order solvers... |
| **SYS-004** | `system_structure` | Higher-order lead c[t+2] without auxiliary variable | `import` | _None_ | 1 warning(s) | Higher-order lead 'c[t+2]' detected. Dyno solvers support le... |
| **SYS-005** | `system_structure` | Empty model file (comments only) | `import` | `LARKParserError` | _None_ | Model description is empty. Provide at least one equation an... |
| **SOL-001** | `solvability` | Blanchard-Kahn condition violation (explosive root with backward variable) | `solve` | `BlanchardKahnError` | _None_ | Blanchard-Kahn condition failed: explosive eigenvalue 2.0 as... |
| **SOL-002** | `solvability` | Singular Jacobian due to collinear equations | `solve` | `ValueError` | _None_ | Jacobian matrix is singular. Equation 2 is a scalar multiple... |

---

## Detailed Diagnostic Profiles

### [SYN-001] Top-level equality used instead of assignment operator
- **Category:** `syntax`
- **User Intent:** User intends to define parameter alpha = 0.35 using Python/Julia syntax.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
alpha = 0.35
k[t] = 0.9 * k[t-1]
k[~] <- 1.0
```
- **Raised Exception:** _None (Execution succeeded or silent NaN)_
- **Warnings Emitted:**
  - `UndefinedSymbolWarning`: Undefined value: alpha at (2, 1). Defaulting to NaN.
  - `UndefinedSymbolWarning`: Undefined parameter(s) used in equation definitions: alpha. Defaulting to NaN.
- **Ideal Diagnostic:** > *Top-level equality detected for 'alpha = 0.35'. In Dyno, '=' defines dynamic equations. Did you mean 'alpha <- 0.35'?*
- **Suggested Fix:** `alpha <- 0.35`

### [SYN-002] Parentheses used for time indexing instead of brackets
- **Category:** `syntax`
- **User Intent:** User writes time subscript as c(t) instead of c[t].
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
alpha <- 0.35
k(t) = 0.9 * k(t-1)
k[~] <- 1.0
```
- **Raised Exception:** `ValueError`: `Undefined function: k`
- **Warnings Emitted:**
  - `UndefinedSymbolWarning`: Undefined value: t at (3, 16). Defaulting to NaN.
- **Ideal Diagnostic:** > *Syntax error: 'k(t)' looks like a time index or lag. Time indices in Dyno must use square brackets: 'k[t]'.*
- **Suggested Fix:** `k[t] = 0.9 * k[t-1]`

### [SYN-003] Trailing semicolon from MATLAB/Dynare convention
- **Category:** `syntax`
- **User Intent:** User ends parameter statement with semicolon.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
alpha <- 0.35;
k[t] = 0.9 * k[t-1]
k[~] <- 1.0
```
- **Raised Exception:** `LARKParserError`: `Unexpected token `;` at (2, 14)`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Semicolons are not required or permitted as statement terminators in Dyno. Remove trailing ';'.*
- **Suggested Fix:** `alpha <- 0.35`

### [SYN-004] Dynare variable declaration block in .dyno file
- **Category:** `syntax`
- **User Intent:** User attempts to declare variables using Dynare syntax.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
var k, c;
parameters alpha;
alpha <- 0.35
k[t] = 0.9 * k[t-1]
k[~] <- 1.0
```
- **Raised Exception:** `LARKParserError`: `Unexpected token `k, c;` at (2, 5)`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Dyno does not require variable declarations. Variables are inferred automatically from time indices [t] and steady-state declarations [~].*
- **Suggested Fix:** `Remove 'var' and 'parameters' declarations.`

### [SYN-005] Dynare implicit lag notation k(-1)
- **Category:** `syntax`
- **User Intent:** User references lagged variable without explicit 't'.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
alpha <- 0.35
k[t] = 0.9 * k(-1)
k[~] <- 1.0
```
- **Raised Exception:** `ValueError`: `Undefined function: k`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Dynare notation 'k(-1)' is not recognized in .dyno files. Use explicit time indexing: 'k[t-1]'.*
- **Suggested Fix:** `k[t] = 0.9 * k[t-1]`

### [SYN-006] Double equals comparison operator in equation
- **Category:** `syntax`
- **User Intent:** User uses '==' instead of '=' for dynamic equation.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
alpha <- 0.35
k[t] == 0.9 * k[t-1]
k[~] <- 1.0
```
- **Raised Exception:** `LARKParserError`: `Unexpected token `= 0.9 * k[t-1]` at (3, 7)`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Dynamic equations in Dyno use a single '=' operator (e.g. 'lhs = rhs').*
- **Suggested Fix:** `k[t] = 0.9 * k[t-1]`

### [SYN-007] Reversed assignment arrow ->
- **Category:** `syntax`
- **User Intent:** User writes assignment with rightward arrow.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
0.35 -> alpha
k[t] = 0.9 * k[t-1]
k[~] <- 1.0
```
- **Raised Exception:** `LARKParserError`: `Unexpected token `> alpha` at (2, 7)`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Assignment in Dyno must be leftward: 'alpha <- 0.35' or 'alpha := 0.35'.*
- **Suggested Fix:** `alpha <- 0.35`

### [SYN-008] Unsupported mathematical function
- **Category:** `syntax`
- **User Intent:** User calls an unsupported function such as sigmoid.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
alpha <- 0.35
k[t] = sigmoid(k[t-1])
k[~] <- 1.0
```
- **Raised Exception:** `ValueError`: `Undefined function: sigmoid`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Function 'sigmoid' is not recognized. Supported functions include: exp, log, sqrt, abs, sin, cos, tan.*
- **Suggested Fix:** `Define sigmoid analytically using exp(x) / (1 + exp(x)).`

### [SYM-001] Typo in parameter name in equation definition
- **Category:** `symbol_resolution`
- **User Intent:** User meant 'alpha' but typed 'alhpa'.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
alpha <- 0.35
k[t] = alhpa * k[t-1]
k[~] <- 1.0
```
- **Raised Exception:** _None (Execution succeeded or silent NaN)_
- **Warnings Emitted:**
  - `UndefinedSymbolWarning`: Undefined value: alhpa at (3, 8). Defaulting to NaN.
  - `UndefinedSymbolWarning`: Undefined parameter(s) used in equation definitions: alhpa. Defaulting to NaN.
- **Ideal Diagnostic:** > *Parameter 'alhpa' used in equation is never defined. Did you mean 'alpha'?*
- **Suggested Fix:** `k[t] = alpha * k[t-1]`

### [SYM-002] Variable in equation without time index
- **Category:** `symbol_resolution`
- **User Intent:** User omits '[t]' when referring to contemporaneous variable c.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
alpha <- 0.35
k[t] = alpha * k[t-1] + c
c[t] = 0.5 * k[t]
k[~] <- 1.0
c[~] <- 0.5
```
- **Raised Exception:** _None (Execution succeeded or silent NaN)_
- **Warnings Emitted:**
  - `UndefinedSymbolWarning`: Undefined value: c at (3, 25). Defaulting to NaN.
  - `UndefinedSymbolWarning`: Undefined parameter(s) used in equation definitions: c. Defaulting to NaN.
- **Ideal Diagnostic:** > *Symbol 'c' appears in equation without a time index [t], but 'c' is declared as a variable. Did you mean 'c[t]'?*
- **Suggested Fix:** `k[t] = alpha * k[t-1] + c[t]`

### [SYM-003] Out-of-order parameter calibration
- **Category:** `symbol_resolution`
- **User Intent:** User references parameter 'b' before its definition.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
a <- b * 2.0
b <- 1.0
k[t] = a * k[t-1]
k[~] <- 1.0
```
- **Raised Exception:** _None (Execution succeeded or silent NaN)_
- **Warnings Emitted:**
  - `UndefinedSymbolWarning`: Undefined value: b at (2, 6). Defaulting to NaN.
  - `UndefinedSymbolWarning`: Undefined parameter(s) used in equation definitions: a. Defaulting to NaN.
- **Ideal Diagnostic:** > *Parameter 'b' was referenced on line 2 before its assignment on line 3.*
- **Suggested Fix:** `Define 'b <- 1.0' before 'a <- b * 2.0'.`

### [SYM-004] Circular parameter calibration
- **Category:** `symbol_resolution`
- **User Intent:** User mutually defines parameters without a base value.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
a <- b + 1.0
b <- a + 1.0
k[t] = a * k[t-1]
k[~] <- 1.0
```
- **Raised Exception:** _None (Execution succeeded or silent NaN)_
- **Warnings Emitted:**
  - `UndefinedSymbolWarning`: Undefined value: b at (2, 6). Defaulting to NaN.
  - `UndefinedSymbolWarning`: Undefined parameter(s) used in equation definitions: a. Defaulting to NaN.
- **Ideal Diagnostic:** > *Circular dependency detected in parameter calibration between 'a' and 'b'.*
- **Suggested Fix:** `Provide an exogenous numerical value for at least one parameter.`

### [SYM-005] Parameter self-reference without initial calibration
- **Category:** `symbol_resolution`
- **User Intent:** User writes recursive parameter expression a <- a + 1.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
a <- a + 1.0
k[t] = a * k[t-1]
k[~] <- 1.0
```
- **Raised Exception:** _None (Execution succeeded or silent NaN)_
- **Warnings Emitted:**
  - `UndefinedSymbolWarning`: Undefined value: a at (2, 6). Defaulting to NaN.
  - `UndefinedSymbolWarning`: Undefined parameter(s) used in equation definitions: a. Defaulting to NaN.
- **Ideal Diagnostic:** > *Parameter 'a' references itself before having an established value.*
- **Suggested Fix:** `Initialize 'a' with a literal number first.`

### [SYM-006] Typo in variable name creating phantom variable
- **Category:** `symbol_resolution`
- **User Intent:** User meant 'k[t]' but typed 'k_cap[t]'.
- **Expected Detection Stage:** `check`
- **Actual Stage Reached:** `check`
- **Source Code Snippet:**
```text
alpha <- 0.35
k_cap[t] = alpha * k[t-1]
k[~] <- 1.0
```
- **Raised Exception:** `UndefinedSymbolError`: `Cannot check model due to uninitialized symbols (variables without steady state: k_cap).`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Variable 'k_cap[t]' appears in equations but has no steady state [~] or transition law.*
- **Suggested Fix:** `k[t] = alpha * k[t-1]`

### [SS-001] Missing steady-state declaration for endogenous variable
- **Category:** `steady_state`
- **User Intent:** User writes dynamic equations but completely omits k[~].
- **Expected Detection Stage:** `check`
- **Actual Stage Reached:** `check`
- **Source Code Snippet:**
```text
alpha <- 0.35
k[t] = alpha * k[t-1]
```
- **Raised Exception:** `UndefinedSymbolError`: `Cannot check model due to uninitialized symbols (variables without steady state: k).`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Variable 'k' has no steady-state declaration. Provide 'k[~] <- ...' or use model.steady() to solve numerically.*
- **Suggested Fix:** `k[~] <- 0.0`

### [SS-002] Typo in steady-state declaration name k_ss
- **Category:** `steady_state`
- **User Intent:** User declares steady state as k_ss <- 1.0 instead of k[~] <- 1.0.
- **Expected Detection Stage:** `check`
- **Actual Stage Reached:** `check`
- **Source Code Snippet:**
```text
alpha <- 0.35
k[t] = alpha * k[t-1]
k_ss <- 1.0
```
- **Raised Exception:** `UndefinedSymbolError`: `Cannot check model due to uninitialized symbols (variables without steady state: k).`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Unused parameter 'k_ss' detected. Did you mean to declare the steady state as 'k[~] <- 1.0'?*
- **Suggested Fix:** `k[~] <- 1.0`

### [SS-003] Inconsistent steady-state value
- **Category:** `steady_state`
- **User Intent:** User supplies an incorrect steady-state guess (residual is non-zero).
- **Expected Detection Stage:** `check`
- **Actual Stage Reached:** `check`
- **Source Code Snippet:**
```text
alpha <- 0.35
k[t] = alpha * k[t-1]
k[~] <- 10.0
```
- **Raised Exception:** `SteadyStateError`: `Steady state values don't satisfy model equations. Max residual is 6.5`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Equation 1 has non-zero steady-state residual (residual = -6.5). The steady-state condition is k = 0, but k[~] was set to 10.0.*
- **Suggested Fix:** `k[~] <- 0.0`

### [SS-004] Steady-state declaration causes division by zero
- **Category:** `steady_state`
- **User Intent:** User sets c[~] = 0 in a model with utility 1/c.
- **Expected Detection Stage:** `check`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
c[~] <- 0.0
1 / c[t] = 1.0
```
- **Raised Exception:** `ZeroDivisionError`: `float division by zero`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Division by zero encountered in equation 1 when evaluated at steady state (c[~] = 0.0).*
- **Suggested Fix:** `c[~] <- 1.0`

### [SS-005] Steady-state evaluation causes negative log domain error
- **Category:** `steady_state`
- **User Intent:** User sets steady state of z to -1 where equation has log(z).
- **Expected Detection Stage:** `check`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
z[~] <- -1.0
y[t] = log(z[t])
```
- **Raised Exception:** `ValueError`: `math domain error`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Domain error: log(z[t]) evaluated with non-positive steady state z[~] = -1.0.*
- **Suggested Fix:** `z[~] <- 1.0`

### [SS-006] Forward reference across steady-state declarations
- **Category:** `steady_state`
- **User Intent:** User defines k[~] referencing y[~] before y[~] is defined.
- **Expected Detection Stage:** `check`
- **Actual Stage Reached:** `check`
- **Source Code Snippet:**
```text
k[~] <- y[~] * 0.5
y[~] <- 2.0
k[t] = 0.5 * k[t-1] + 0.5 * y[t]
y[t] = 2.0
```
- **Raised Exception:** `UndefinedSymbolError`: `Cannot check model due to uninitialized symbols (variables without steady state: k).`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Steady state 'y[~]' was referenced in 'k[~]' before 'y[~]' was assigned.*
- **Suggested Fix:** `Define 'y[~] <- 2.0' before 'k[~] <- y[~] * 0.5'.`

### [SHK-001] Shock variable in equations without distribution declaration
- **Category:** `shocks`
- **User Intent:** User uses e[t] in equation but never writes e[t] <- N(0, sigma^2).
- **Expected Detection Stage:** `solve`
- **Actual Stage Reached:** `check`
- **Source Code Snippet:**
```text
rho <- 0.9
z[t] = rho * z[t-1] + e[t]
z[~] <- 0.0
```
- **Raised Exception:** `UndefinedSymbolError`: `Cannot check model due to uninitialized symbols (variables without steady state: e).`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Variable 'e' appears to be an exogenous shock but has no distribution. Declare 'e[t] <- N(0, sigma^2)'.*
- **Suggested Fix:** `e[t] <- N(0, 0.01^2)`

### [SHK-002] Shock distribution too many arguments
- **Category:** `shocks`
- **User Intent:** User calls N(0, 0.01, 1) with three arguments instead of std or mean and std.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
rho <- 0.9
z[t] = rho * z[t-1] + e[t]
z[~] <- 0.0
e[t] <- N(0, 0.01, 1)
```
- **Raised Exception:** `TypeError`: `N() takes 1 or 2 arguments: N(std) or N(mean, std), but 3 were given`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *N(...) requires 1 or 2 arguments: N(std) or N(mean, std). E.g. 'e[t] <- N(0.01)'.*
- **Suggested Fix:** `e[t] <- N(0.01)`

### [SHK-003] Shock assigned as parameter without [t]
- **Category:** `shocks`
- **User Intent:** User writes e <- N(0, 0.01) instead of e[t] <- N(0, 0.01).
- **Expected Detection Stage:** `solve`
- **Actual Stage Reached:** `check`
- **Source Code Snippet:**
```text
rho <- 0.9
z[t] = rho * z[t-1] + e[t]
z[~] <- 0.0
e <- N(0, 0.01)
```
- **Raised Exception:** `UndefinedSymbolError`: `Cannot check model due to uninitialized symbols (variables without steady state: e).`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Shock process assigned to constant 'e' rather than time series 'e[t]'. Use 'e[t] <- N(...)'.*
- **Suggested Fix:** `e[t] <- N(0, 0.01)`

### [SHK-004] Shock distribution function name typo Normal()
- **Category:** `shocks`
- **User Intent:** User writes Normal(0, 0.01) instead of N(0, 0.01).
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
rho <- 0.9
z[t] = rho * z[t-1] + e[t]
z[~] <- 0.0
e[t] <- Normal(0, 0.01)
```
- **Raised Exception:** `ValueError`: `Undefined function: Normal`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Distribution 'Normal' is not recognized. Use 'N(std)' or 'N(mean, std)' for Gaussian shocks.*
- **Suggested Fix:** `e[t] <- N(0.01)`

### [SYS-001] Underdetermined model (fewer equations than endogenous variables)
- **Category:** `system_structure`
- **User Intent:** User forgets one equation in a 2-variable model.
- **Expected Detection Stage:** `solve`
- **Actual Stage Reached:** `check`
- **Source Code Snippet:**
```text
alpha <- 0.35
k[t] = alpha * k[t-1] + c[t] + e[t]
k[~] <- 1.0
c[~] <- 0.5
e[t] <- N(0, 0.01)
```
- **Raised Exception:** `SteadyStateError`: `Steady state values don't satisfy model equations. Max residual is 0.15000000000000002`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Model is underdetermined: 1 equation provided for 2 endogenous variables ('k', 'c'). Provide an equation for 'c[t]'.*
- **Suggested Fix:** `Add equation for c[t].`

### [SYS-002] Overdetermined model (more equations than endogenous variables)
- **Category:** `system_structure`
- **User Intent:** User accidentally duplicates an equation.
- **Expected Detection Stage:** `solve`
- **Actual Stage Reached:** `steady`
- **Source Code Snippet:**
```text
alpha <- 0.35
k[t] = alpha * k[t-1] + e[t]
k[t] = 0.8 * k[t-1] + e[t]
k[~] <- 0.0
e[t] <- N(0, 0.01)
```
- **Raised Exception:** `ValueError`: `The array returned by a function changed size between calls`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Model is overdetermined: 2 equations provided for 1 endogenous variable ('k'). Remove redundant equation.*
- **Suggested Fix:** `Remove duplicated or conflicting equation.`

### [SYS-003] Higher-order lag k[t-2] without auxiliary state
- **Category:** `system_structure`
- **User Intent:** User models AR(2) process directly with lag 2.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
alpha1 <- 0.5
alpha2 <- 0.2
k[t] = alpha1 * k[t-1] + alpha2 * k[t-2] + e[t]
k[~] <- 0.0
e[t] <- N(0, 0.01)
```
- **Raised Exception:** _None (Execution succeeded or silent NaN)_
- **Warnings Emitted:**
  - `UndefinedSymbolWarning`: Higher-order lead/lag detected (k[t-2]). Dyno solvers currently support shifts in [-1, 1].
- **Ideal Diagnostic:** > *Higher-order lag 'k[t-2]' detected. Dyno first-order solvers support leads and lags within [-1, 1]. Introduce an auxiliary variable 'k_lag[t] = k[t-1]'.*
- **Suggested Fix:** `k_lag[t] = k[t-1]
k[t] = alpha1 * k[t-1] + alpha2 * k_lag[t-1]`

### [SYS-004] Higher-order lead c[t+2] without auxiliary variable
- **Category:** `system_structure`
- **User Intent:** User models two-period ahead forward expectation.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
beta <- 0.95
c[t] = beta * c[t+2] + e[t]
c[~] <- 0.0
e[t] <- N(0, 0.01)
```
- **Raised Exception:** _None (Execution succeeded or silent NaN)_
- **Warnings Emitted:**
  - `UndefinedSymbolWarning`: Higher-order lead/lag detected (c[t+2]). Dyno solvers currently support shifts in [-1, 1].
- **Ideal Diagnostic:** > *Higher-order lead 'c[t+2]' detected. Dyno solvers support leads within [-1, 1]. Define an auxiliary expected value variable.*
- **Suggested Fix:** `c_lead[t] = c[t+1]
c[t] = beta * c_lead[t+1] + e[t]`

### [SYS-005] Empty model file (comments only)
- **Category:** `system_structure`
- **User Intent:** User loads an empty model file.
- **Expected Detection Stage:** `import`
- **Actual Stage Reached:** `import`
- **Source Code Snippet:**
```text
# Only comments in this model file
# Nothing else
```
- **Raised Exception:** `LARKParserError`: `Unexpected token `` at (1, 1)`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Model description is empty. Provide at least one equation and parameter definition.*
- **Suggested Fix:** `Add equations and parameters.`

### [SOL-001] Blanchard-Kahn condition violation (explosive root with backward variable)
- **Category:** `solvability`
- **User Intent:** User specifies an unstable backward-looking autoregressive system.
- **Expected Detection Stage:** `solve`
- **Actual Stage Reached:** `solve`
- **Source Code Snippet:**
```text
k[t] = 2.0 * k[t-1] + e[t]
k[~] <- 0.0
e[t] <- N(0, 0.01)
```
- **Raised Exception:** `BlanchardKahnError`: `Eigenvalue condition not satisfied: l_(n)=2.0, l_(n+1)=inf. No stable solution.`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Blanchard-Kahn condition failed: explosive eigenvalue 2.0 associated with predetermined backward-looking variable 'k'. No stable path exists.*
- **Suggested Fix:** `Ensure coefficient on k[t-1] has absolute value < 1.`

### [SOL-002] Singular Jacobian due to collinear equations
- **Category:** `solvability`
- **User Intent:** User writes two linearly dependent equations.
- **Expected Detection Stage:** `solve`
- **Actual Stage Reached:** `solve`
- **Source Code Snippet:**
```text
k[t] = 0.5 * k[t-1] + c[t]
2 * k[t] = k[t-1] + 2 * c[t]
k[~] <- 0.0
c[~] <- 0.0
```
- **Raised Exception:** `ValueError`: `could not broadcast input array from shape (2,) into shape (4,)`
- **Warnings Emitted:** _None_
- **Ideal Diagnostic:** > *Jacobian matrix is singular. Equation 2 is a scalar multiple of Equation 1. Replace with an independent equilibrium condition.*
- **Suggested Fix:** `Replace redundant equation with an independent relation.`
