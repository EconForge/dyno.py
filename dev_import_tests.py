import dyno
from rich import print, inspect
import yaml

from dyno.symbolic_model import SymbolicModel


data = yaml.safe_load(open("import_tests.yaml"))

model = SymbolicModel(txt=data['solvable']['rbc_deterministic'])

model


# from dyno.solver import solve as perturb
# r, A,B,C,D = model.jacobians
# X, evs = perturb(A,B,C)
# evs.real


from dyno.solver import deterministic_solve
import time
t0 = time.time()
sol = deterministic_solve(model, verbose=True, T=250, maxit=20)
t1 = time.time()
print("Elapsed time:", t1 - t0)
print(sol)





import numpy as np
v0 = model.deterministic_guess(T= 12)

T = v0.shape[0] - 1
u0 = np.array(v0).ravel()

from dyno.solver import newton
verbose=True
fobj = lambda u: model.deterministic_residuals_with_jacobian(u, sparsify=True)
fobj(u0)

from scipy.sparse.linalg import spsolve
from scipy.sparse import linalg as linalg
model.deterministic_residuals_with_jacobian(u0, sparsify=True)

u0 = v0.ravel().copy()
u1 = u0.copy()

a0, J0  = fobj(u0)
delta = spsolve(J0, a0)

u1 = u0 - delta/16*0.0
print(f"Residuals: {abs(fobj(u1)[0]).max()}" )


from numpy.linalg import solve
r0 = np.random.randn(8)



np.linalg.matrix_rank(J[:32,:32])

# Matrix J0 is not invertible

def analyze_singular_matrix(J, tolerance=1e-10):
    """
    Analyze a singular sparse matrix to identify problematic rows/columns.
    
    Parameters:
    -----------
    J : scipy.sparse matrix or numpy array
        The matrix to analyze
    tolerance : float
        Threshold for considering values as zero
        
    Returns:
    --------
    dict with:
        - zero_rows: rows that are all zeros
        - zero_cols: columns that are all zeros
        - near_zero_rows: rows with very small norm
        - near_zero_cols: columns with very small norm
        - rank: matrix rank
        - condition_number: condition number (if computable)
    """
    import numpy as np
    from scipy.sparse import issparse
    
    n_rows, n_cols = J.shape
    
    # Convert to dense for rank computation (if small enough)
    if issparse(J):
        J_dense = J.toarray()
        J_csr = J.tocsr()
        J_csc = J.tocsc()
        nnz = J.nnz
    else:
        J_dense = J
        J_csr = J
        J_csc = J
        nnz = np.count_nonzero(J)
    
    # Compute row norms efficiently
    if issparse(J):
        row_norms = np.array([np.linalg.norm(J_csr[i].toarray()) for i in range(n_rows)])
        col_norms = np.array([np.linalg.norm(J_csc[:, j].toarray()) for j in range(n_cols)])
    else:
        row_norms = np.linalg.norm(J, axis=1)
        col_norms = np.linalg.norm(J, axis=0)
    
    # Find zero and near-zero rows/columns
    zero_rows = np.where(row_norms == 0)[0]
    near_zero_rows = np.where((row_norms > 0) & (row_norms < tolerance))[0]
    zero_cols = np.where(col_norms == 0)[0]
    near_zero_cols = np.where((col_norms > 0) & (col_norms < tolerance))[0]
    
    # Compute rank using numpy.linalg.matrix_rank
    try:
        rank = np.linalg.matrix_rank(J_dense, tol=tolerance)
    except Exception as e:
        print(f"Could not compute rank: {e}")
        rank = None
    
    # Compute condition number
    try:
        cond = np.linalg.cond(J_dense)
    except Exception as e:
        print(f"Could not compute condition number: {e}")
        cond = None
    
    results = {
        'zero_rows': zero_rows,
        'zero_cols': zero_cols,
        'near_zero_rows': near_zero_rows,
        'near_zero_cols': near_zero_cols,
        'row_norms': row_norms,
        'col_norms': col_norms,
        'rank': rank,
        'condition_number': cond,
        'shape': J.shape,
        'nnz': nnz
    }
    
    return results

def print_analysis(results):
    """Pretty print the analysis results"""
    print("\n" + "="*60)
    print("MATRIX SINGULARITY ANALYSIS")
    print("="*60)
    print(f"Matrix shape: {results['shape']}")
    print(f"Non-zero elements: {results['nnz']}")
    print(f"Sparsity: {100 * (1 - results['nnz']/(results['shape'][0]*results['shape'][1])):.2f}%")
    
    if results['rank'] is not None:
        print(f"\nMatrix rank: {results['rank']}")
        print(f"Expected rank (full rank): {min(results['shape'])}")
        print(f"Rank deficiency: {min(results['shape']) - results['rank']}")
    
    if results['condition_number'] is not None:
        print(f"Condition number: {results['condition_number']:.2e}")
    
    print(f"\nZero rows: {len(results['zero_rows'])}")
    if len(results['zero_rows']) > 0 and len(results['zero_rows']) <= 20:
        print(f"  Indices: {results['zero_rows']}")
    
    print(f"Near-zero rows (norm < 1e-10): {len(results['near_zero_rows'])}")
    if len(results['near_zero_rows']) > 0 and len(results['near_zero_rows']) <= 20:
        print(f"  Indices: {results['near_zero_rows']}")
    
    print(f"\nZero columns: {len(results['zero_cols'])}")
    if len(results['zero_cols']) > 0 and len(results['zero_cols']) <= 20:
        print(f"  Indices: {results['zero_cols']}")
    
    print(f"Near-zero columns (norm < 1e-10): {len(results['near_zero_cols'])}")
    if len(results['near_zero_cols']) > 0 and len(results['near_zero_cols']) <= 20:
        print(f"  Indices: {results['near_zero_cols']}")
    
    # Show row/column norms statistics
    print(f"\nRow norms - Min: {results['row_norms'].min():.2e}, Max: {results['row_norms'].max():.2e}")
    print(f"Column norms - Min: {results['col_norms'].min():.2e}, Max: {results['col_norms'].max():.2e}")
    
    print("="*60)

# Analyze the matrix J0
analysis = analyze_singular_matrix(J0)
print_analysis(analysis)

def suggest_removals(J, tolerance=1e-10):
    """
    Suggest which rows/columns to remove based on singularity analysis.
    
    Parameters:
    -----------
    J : scipy.sparse matrix
        The sparse matrix to analyze
    tolerance : float
        Threshold for considering a row/column as near-zero
        
    Returns:
    --------
    dict with suggested rows and columns to remove
    """
    import numpy as np
    from scipy.sparse import issparse
    
    analysis = analyze_singular_matrix(J, tolerance=tolerance)
    
    # Priority 1: Remove exact zero rows/columns
    remove_rows = set(analysis['zero_rows'].tolist())
    remove_cols = set(analysis['zero_cols'].tolist())
    
    # Priority 2: Remove near-zero rows/columns
    remove_rows.update(analysis['near_zero_rows'].tolist())
    remove_cols.update(analysis['near_zero_cols'].tolist())
    
    # Priority 3: Find linearly dependent columns
    linear_dependencies = []
    
    if analysis['rank'] is not None and analysis['rank'] < min(J.shape):
        # Convert to dense for analysis
        if issparse(J):
            J_dense = J.toarray()
        else:
            J_dense = J
        
        n_cols = J_dense.shape[1]
        
        # Use QR decomposition with column pivoting to find dependent columns
        try:
            from scipy.linalg import qr
            
            # QR with column pivoting - returns (Q, R, P) when pivoting=True
            qr_result = qr(J_dense, pivoting=True, mode='economic')
            
            # Handle the return value (Q, R, P)
            if len(qr_result) == 3:
                Q, R, P = qr_result
            else:
                raise ValueError(f"Expected 3 return values from QR, got {len(qr_result)}")
            
            # Find columns where diagonal of R is near zero (dependent columns)
            r_diag = np.abs(np.diag(R))
            dependent_cols = np.where(r_diag < tolerance)[0]
            
            # For each dependent column, find which columns it depends on
            for dep_col_idx in dependent_cols:
                if dep_col_idx < len(r_diag):
                    # Original column index (before pivoting)
                    orig_col = P[dep_col_idx]
                    
                    # Find which columns this depends on (non-zero entries in R)
                    if dep_col_idx < R.shape[0]:
                        # Get the row in R corresponding to this column
                        r_row = R[dep_col_idx, :]
                        depends_on = []
                        coeffs = []
                        
                        # Look at columns to the left (independent columns)
                        for i in range(dep_col_idx):
                            if abs(r_row[i]) > tolerance:
                                depends_on.append(int(P[i]))
                                coeffs.append(float(r_row[i] / r_diag[i]))
                        
                        if depends_on:
                            linear_dependencies.append({
                                'column': int(orig_col),
                                'depends_on': depends_on,
                                'coefficients': coeffs
                            })
                            remove_cols.add(orig_col)
        
        except Exception as e:
            print(f"Could not perform QR decomposition: {e}")
            # Fallback: simple column correlation check
            try:
                for i in range(n_cols):
                    if i in remove_cols:
                        continue
                    for j in range(i + 1, n_cols):
                        if j in remove_cols:
                            continue
                        # Check if columns i and j are linearly dependent
                        col_i = J_dense[:, i]
                        col_j = J_dense[:, j]
                        
                        # Skip if either column is zero
                        norm_i = np.linalg.norm(col_i)
                        norm_j = np.linalg.norm(col_j)
                        if norm_i < tolerance or norm_j < tolerance:
                            continue
                        
                        # Check correlation
                        correlation = np.abs(np.dot(col_i, col_j)) / (norm_i * norm_j)
                        if correlation > 1 - tolerance:
                            # Columns are nearly parallel
                            linear_dependencies.append({
                                'column': j,
                                'depends_on': [i],
                                'coefficients': [np.dot(col_i, col_j) / np.dot(col_i, col_i)]
                            })
                            remove_cols.add(j)
            except Exception as e2:
                print(f"Fallback correlation check also failed: {e2}")
    
    suggestions = {
        'remove_rows': sorted(remove_rows),
        'remove_cols': sorted(remove_cols),
        'linear_dependencies': linear_dependencies,
        'analysis': analysis
    }
    
    print("\n" + "="*60)
    print("REMOVAL SUGGESTIONS")
    print("="*60)
    print(f"Suggested rows to remove: {len(suggestions['remove_rows'])}")
    if len(suggestions['remove_rows']) <= 20:
        print(f"  Indices: {suggestions['remove_rows']}")
    
    print(f"\nSuggested columns to remove: {len(suggestions['remove_cols'])}")
    if len(suggestions['remove_cols']) <= 20:
        print(f"  Indices: {suggestions['remove_cols']}")
    
    if linear_dependencies:
        print(f"\nLinear dependencies found: {len(linear_dependencies)}")
        for dep in linear_dependencies[:10]:  # Show first 10
            print(f"  Column {dep['column']} is a linear combination of columns {dep['depends_on']}")
            if len(dep['coefficients']) <= 5:
                print(f"    Coefficients: {[f'{c:.4f}' for c in dep['coefficients']]}")
    
    print("="*60)
    
    return suggestions

# Get suggestions
suggestions = suggest_removals(J0.todense())


J = (J0.todense())
for i in range(J0.shape[0]):

    M = np.vstack([J[0:i,:], J[i+1:J.shape[0],:]])
    print(f"Removing row {i}, new rank: {np.linalg.matrix_rank(M)}")




from matplotlib import pyplot as plt
plt.imshow(np.corrcoef(J0.toarray().T))



res, nit = newton(
    u0,
    jactype="sparse",
    verbose=verbose,
)

w0 = res.reshape(v0.shape)


# for key in data['valid']:
    
#     txt = data['valid'][key]
#     model = SymbolicModel(txt=txt)
#     print(f"--- Model: {key} ---")
#     print(model.context)
#     print(model.equations)
#     print(model.residuals)


# for key in data['solvable']:

#     txt = data['solvable'][key]
#     model = SymbolicModel(txt=txt)
#     print(f"--- Model: {key} ---")
#     # print(model.context)
#     # print(model.equations)
#     # print(model.residuals)
#     inspect(model.processes)
#     print(model.checks)
#     print(model.solve())