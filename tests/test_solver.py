def test_all(tol=1e-9):
    import dyno
    from .solver import solve, solve_qz
    import numpy as np
    
    n  = 4
    A0 = np.eye(n)
    B0 = np.diag(30 * np.ones(n))
    B0[0, 0] = 20
    B0[-1, -1] = 20
    B0 = B0 - np.diag(10 * np.ones(n - 1), -1)
    B0 = B0 - np.diag(10 * np.ones(n - 1), 1)
    C0 = np.diag(15 * np.ones(n))
    C0 = C0 - np.diag(5 * np.ones(n - 1), -1)
    C0 = C0 - np.diag(5 * np.ones(n - 1), 1)
    
    # Check QZ decomposition solver
    X0_qz = solve_qz(A0, B0, C0)
    res_qz = A0 @ X0_qz @ X0_qz + B0 @ X0_qz + C0
    err_qz = np.linalg.norm(res_qz, 1)
    assert err_qz < tol, "QZ method did not find a solution (residual = {}).".format(err_qz)
    
    
    # Check Time Iteration solver
    X0_ti = solve(A0, B0, C0)
    res_ti = A0 @ X0_ti @ X0_ti + B0 @ X0_ti + C0
    err_ti = np.linalg.norm(res_ti, 1)
    assert err_ti < tol, "Time Iteration did not find a solution (residual = {}).".format(err_ti)
    
    