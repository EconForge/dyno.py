def test_all(tol=1e-9):

    import dyno
    from dyno.solver import solve_qz, solve_ti
    import numpy as np
    
    n  = 4
    A0 = np.eye(n) + np.random.random((n,n)) * 0.01
    B0 = np.diag(30 * np.ones(n))
    B0[0, 0] = 20
    B0[-1, -1] = 20
    B0 = B0 - np.diag(10 * np.ones(n - 1), -1)
    B0 = B0 - np.diag(10 * np.ones(n - 1), 1)  + np.random.random((n,n)) * 0.01
    C0 = np.diag(15 * np.ones(n))
    C0 = C0 - np.diag(5 * np.ones(n - 1), -1)
    C0 = C0 - np.diag(5 * np.ones(n - 1), 1)  + np.random.random((n,n)) * 0.01
    
    import time
    # Check QZ decomposition solver
    t1 = time.time()
    X0_qz = solve_qz(A0, B0, C0)
    t2 = time.time()
    print(f"Elapsed: {t2-t1}")
    res_qz = A0 @ X0_qz @ X0_qz + B0 @ X0_qz + C0
    err_qz = np.linalg.norm(res_qz, 1)
    assert err_qz < tol, "QZ method did not find a solution (residual = {}).".format(err_qz)
    
    
    # Check Time Iteration solver
    t1 = time.time()
    X0_ti = solve_ti(A0, B0, C0)
    t2 = time.time()
    print(f"Elapsed: {t2 -t1}")
    res_ti = A0 @ X0_ti @ X0_ti + B0 @ X0_ti + C0
    err_ti = np.linalg.norm(res_ti, 1)
    assert err_ti < tol, "Time Iteration did not find a solution (residual = {}).".format(err_ti)
    
