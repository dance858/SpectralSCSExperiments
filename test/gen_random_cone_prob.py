import numpy as np
from scipy import sparse
from time import time
import pdb

#############################################
#      Generate random cone problems        #
#############################################


def gen_feasible(K, n, density):
    m = get_scs_cone_dims(K)
    z = np.random.randn(m)
    y = proj_dual_cone(z, K)  # y = s - z;
    s = y - z  # s = proj_cone(z,K)

    A = sparse.rand(m, n, density, format="csc")
    A.data = np.random.randn(A.nnz)
    x = np.random.randn(n)
    c = -np.transpose(A).dot(y)
    b = A.dot(x) + s

    data = {"A": A, "b": b, "c": c}
    return data, np.dot(c, x)


def gen_infeasible(K, n):
    m = get_scs_cone_dims(K)

    z = np.random.randn(m)
    y = proj_dual_cone(z, K)  # y = s - z;
    A = np.random.randn(m, n)
    A = (
        A - np.outer(y, np.transpose(A).dot(y)) / np.linalg.norm(y) ** 2
    )  # dense...

    b = np.random.randn(m)
    b = -b / np.dot(b, y)

    data = {"A": sparse.csc_matrix(A), "b": b, "c": np.random.randn(n)}
    return data


def gen_unbounded(K, n):
    m = get_scs_cone_dims(K)

    z = np.random.randn(m)
    s = proj_cone(z, K)
    A = np.random.randn(m, n)
    x = np.random.randn(n)
    A = A - np.outer(s + A.dot(x), x) / np.linalg.norm(x) ** 2
    # dense...
    c = np.random.randn(n)
    c = -c / np.dot(c, x)

    data = {"A": sparse.csc_matrix(A), "b": np.random.randn(m), "c": c}
    return data


def pos(x):
    return (x + abs(x)) / 2


def get_scs_cone_dims(K):
    l = K["z"] + K["l"]
    for i in range(0, len(K["q"])):
        l = l + K["q"][i]

    for i in range(0, len(K["s"])):
        l = l + get_sd_cone_size(K["s"][i])
    
    for i in range(0, len(K["d"])):
        l = l + get_sd_cone_size(K["d"][i]) + 2

    for i in range(0, len(K["nuc_m"])):
        l = l + K["nuc_m"][i] * K["nuc_n"][i] + 1

    l = l + K["ep"] * 3
    l = l + K["ed"] * 3
    l = l + len(K["p"]) * 3
    return int(l)


def proj_dual_cone(z, c):
    return z + proj_cone(-z, c)


def get_sd_cone_size(n):
    return int((n * (n + 1)) / 2)


def proj_cone(z, c):
    z = np.copy(z)
    free_len = c["z"]
    lp_len = c["l"]
    q = c["q"]
    s = c["s"]
    p = c["p"]
    d = c["d"]
    nuc_m = c["nuc_m"]
    nuc_n = c["nuc_n"]
    # free/zero cone
    z[0:free_len] = 0
    # lp cone
    z[free_len : lp_len + free_len] = pos(z[free_len : lp_len + free_len])
    # SOCs
    idx = lp_len + free_len
    for i in range(0, len(q)):
        z[idx : idx + q[i]] = proj_soc(z[idx : idx + q[i]])
        idx = idx + q[i]
    # SDCs
    for i in range(0, len(s)):
        sz = get_sd_cone_size(s[i])
        z[idx : idx + sz] = proj_sdp(z[idx : idx + sz], s[i])
        idx = idx + sz
    # Exp primal
    for i in range(0, c["ep"]):
        z[idx : idx + 3] = project_exp_bisection(z[idx : idx + 3])
        idx = idx + 3
    # Exp dual
    for i in range(0, c["ed"]):
        z[idx : idx + 3] = z[idx : idx + 3] + project_exp_bisection(
            -z[idx : idx + 3]
        )
        idx = idx + 3
    # Power
    for i in range(0, len(p)):
        if p[i] >= 0:  # primal
            z[idx : idx + 3] = proj_pow(z[idx : idx + 3], p[i])
        else:  # dual
            z[idx : idx + 3] = z[idx : idx + 3] + proj_pow(
                -z[idx : idx + 3], -p[i]
            )
        idx = idx + 3
    # logdet cone
    for i in range(0, len(d)):
        sz = get_sd_cone_size(d[i])
        ZMatrix = unvectorize(z[idx + 2: idx + 2 + sz], d[i])
        projTPython, projVPython, projXPythonVec = logDetConeProjPy(z[idx], z[idx + 1], ZMatrix, d[i])
        z[idx] = projTPython
        z[idx + 1] = projVPython
        z[idx + 2: idx + 2 + sz] = projXPythonVec
        idx = idx + 2 + sz
    # TODO: nuclear norm cone
    for i in range(0, len(nuc_m)):
        sz = nuc_m[i] * nuc_n[i]
        ZMatrix = z[idx + 1: idx + 1 + sz].reshape((nuc_m[i], nuc_n[i]), order='F')
        proj1, proj2 = nuclearConeProjPy(z[idx], ZMatrix, nuc_m[i], nuc_n[i])
        z[idx] = proj1 
        z[idx+1:idx+1+sz] = proj2.flatten(order='F')
        idx = idx + 1 + sz
    return z


def proj_soc(tt):
    tt = np.copy(tt)
    if len(tt) == 0:
        return
    elif len(tt) == 1:
        return pos(tt)

    v1 = tt[0]
    v2 = tt[1:]
    if np.linalg.norm(v2) <= -v1:
        v2 = np.zeros(len(v2))
        v1 = 0
    elif np.linalg.norm(v2) > abs(v1):
        v2 = 0.5 * (1 + v1 / np.linalg.norm(v2)) * v2
        v1 = np.linalg.norm(v2)
    tt[0] = v1
    tt[1:] = v2
    return tt


def proj_sdp(z, n):
    z = np.copy(z)
    if n == 0:
        return
    elif n == 1:
        return pos(z)
    tidx = np.triu_indices(n)
    tidx = (tidx[1], tidx[0])
    didx = np.diag_indices(n)

    a = np.zeros((n, n))
    a[tidx] = z
    a = a + np.transpose(a)
    a[didx] = a[didx] / np.sqrt(2.0)

    w, v = np.linalg.eig(a)  # cols of v are eigenvectors
    w = pos(w)
    a = np.dot(v, np.dot(np.diag(w), np.transpose(v)))
    a[didx] = a[didx] / np.sqrt(2.0)
    z = a[tidx]
    return np.real(z)


def proj_pow(v, a):
    CONE_MAX_ITERS = 20
    CONE_TOL = 1e-8

    if v[0] >= 0 and v[1] >= 0 and (v[0] ** a) * (v[1] ** (1 - a)) >= abs(v[2]):
        return v

    if (
        v[0] <= 0
        and v[1] <= 0
        and ((-v[0] / a) ** a) * ((-v[1] / (1 - a)) ** (1 - a)) >= abs(v[2])
    ):
        return np.zeros(
            3,
        )

    xh = v[0]
    yh = v[1]
    zh = v[2]
    rh = abs(zh)
    r = rh / 2
    for iter in range(0, CONE_MAX_ITERS):
        x = calc_x(r, xh, rh, a)
        y = calc_x(r, yh, rh, 1 - a)

        f = calc_f(x, y, r, a)
        if abs(f) < CONE_TOL:
            break

        dxdr = calcdxdr(x, xh, rh, r, a)
        dydr = calcdxdr(y, yh, rh, r, (1 - a))
        fp = calc_fp(x, y, dxdr, dydr, a)

        r = min(max(r - f / fp, 0), rh)

    z = np.sign(zh) * r
    v[0] = x
    v[1] = y
    v[2] = z
    return v


def calc_x(r, xh, rh, a):
    return max(0.5 * (xh + np.sqrt(xh * xh + 4 * a * (rh - r) * r)), 1e-12)


def calcdxdr(x, xh, rh, r, a):
    return a * (rh - 2 * r) / (2 * x - xh)


def calc_f(x, y, r, a):
    return (x**a) * (y ** (1 - a)) - r


def calc_fp(x, y, dxdr, dydr, a):
    return (x**a) * (y ** (1 - a)) * (a * dxdr / x + (1 - a) * dydr / y) - 1


def project_exp_bisection(v):
    v = np.copy(v)
    r = v[0]
    s = v[1]
    t = v[2]
    # v in cl(Kexp)
    if (s > 0 and t > 0 and r <= s * np.log(t / s)) or (
        r <= 0 and s == 0 and t >= 0
    ):
        return v
    # -v in Kexp^*
    if (-r < 0 and r * np.exp(s / r) <= -np.exp(1) * t) or (
        -r == 0 and -s >= 0 and -t >= 0
    ):
        return np.zeros(
            3,
        )
    # special case with analytical solution
    if r < 0 and s < 0:
        v[1] = 0
        v[2] = max(v[2], 0)
        return v

    x = np.copy(v)
    ub, lb = get_rho_ub(v)
    for iter in range(0, 100):
        rho = (ub + lb) / 2
        g, x = calc_grad(v, rho, x)
        if g > 0:
            lb = rho
        else:
            ub = rho
        if ub - lb < 1e-6:
            break
    return x


def get_rho_ub(v):
    lb = 0
    rho = 2 ** (-3)
    g, z = calc_grad(v, rho, v)
    while g > 0:
        lb = rho
        rho = rho * 2
        g, z = calc_grad(v, rho, z)
    ub = rho
    return ub, lb


def calc_grad(v, rho, warm_start):
    x = solve_with_rho(v, rho, warm_start[1])
    if x[1] == 0:
        g = x[0]
    else:
        g = x[0] + x[1] * np.log(x[1] / x[2])
    return g, x


def solve_with_rho(v, rho, w):
    x = np.zeros(3)
    x[2] = newton_exp_onz(rho, v[1], v[2], w)
    x[1] = (1 / rho) * (x[2] - v[2]) * x[2]
    x[0] = v[0] - rho
    return x


def newton_exp_onz(rho, y_hat, z_hat, w):
    t = max(max(w - z_hat, -z_hat), 1e-6)
    for iter in range(0, 100):
        f = (1 / rho**2) * t * (t + z_hat) - y_hat / rho + np.log(t / rho) + 1
        fp = (1 / rho**2) * (2 * t + z_hat) + 1 / t

        t = t - f / fp
        if t <= -z_hat:
            t = -z_hat
            break
        elif t <= 0:
            t = 0
            break
        elif abs(f) < 1e-6:
            break
    return t + z_hat


# --------------------------------------------------
# Code for projecting onto the log determinant cone
# --------------------------------------------------
def stack_lower_triangular_columns(matrix):
    assert np.allclose(matrix, matrix.T), "Matrix must be symmetric"
    lower_triangular = np.tril(matrix)
    stacked_columns = lower_triangular.T.flatten()
    stacked_columns = stacked_columns[stacked_columns != 0]
    return stacked_columns

def scale_off_diagonal(matrix, scale_factor):
    assert np.allclose(matrix, matrix.T), "Matrix must be symmetric"
    diag = np.diag(matrix)
    off_diag_scaled = matrix.copy()
    np.fill_diagonal(off_diag_scaled, 0)  
    off_diag_scaled *= scale_factor
    scaled_matrix = off_diag_scaled + np.diag(diag)
    return scaled_matrix

def vectorize(matrix):
    return stack_lower_triangular_columns(scale_off_diagonal(matrix, np.sqrt(2)))

def unstack_lower_triangular_columns(stacked_columns, n):
    lower_triangular = np.zeros((n, n))
    index = 0
    for col in range(n):
        for row in range(col, n):
            lower_triangular[row, col] = stacked_columns[index]
            index += 1
    
    return lower_triangular

def unvectorize(stacked_columns, n):
    lower_triangular = unstack_lower_triangular_columns(stacked_columns, n)
    diag = np.diag(lower_triangular)
    unscaled_lower_triangular = lower_triangular / np.sqrt(2)
    np.fill_diagonal(unscaled_lower_triangular, diag)
    symmetric_matrix = unscaled_lower_triangular + unscaled_lower_triangular.T - np.diag(diag)
    return symmetric_matrix

def logDetConeProjPy(t0, v0, X0Matrix, n):
    x0, evecs = np.linalg.eigh(X0Matrix)
    sqrt2 = np.sqrt(2)
    solLogConeNewton, _ = NewtonProjLogCone(sqrt2*t0, sqrt2*v0, sqrt2*x0.reshape((n, 1)), n) 
    solLogConeNewton = np.squeeze(solLogConeNewton)
    projTPython = 1/sqrt2 * solLogConeNewton[0]
    projVPython = 1/sqrt2 * solLogConeNewton[1]
    projXPython = 1/sqrt2 * evecs @ np.diag(solLogConeNewton[2:]) @ evecs.T
    projXPythonVec = vectorize(projXPython)
    return projTPython, projVPython, projXPythonVec

LINESEARCH_RELATIVE_TOL = 1e-15
MIN_X = 1e-14
MIN_V = 1e-6
MIN_FLOAT = 1e-15
TOL_ABSOLUTE_DIFF_INF = 1e-12


def objVal(var, tBar, vBar, xBar, n):
     v = var[0, 0]
     x = var[1:n+1]
     assert(x.shape == xBar.shape)
     assert(v > 0)
     assert(np.all(x) > 0)
     sx = -(v * np.sum(np.log(x)) - n * v * np.log(v))
     y = 0.5 * np.square(sx - tBar) + \
         0.5 * np.square(v - vBar) + 0.5 * np.square(np.linalg.norm(x - xBar))
     return y


def NewtonProjLogCone(tBar, vBar, xBar, n, tol = 1e-10, max_iter = 100, 
               alpha = 0.05, beta = 0.8):
    
    start = time()
    stats = {}
    # return if (tBar, vBar, xBar) belongs to cone
    if (vBar > 0 and np.all(xBar > 0) 
        and (-vBar * np.sum(np.log(xBar/vBar)) <= tBar)):
        print("belongs to cone")
        return np.concatenate([np.array([[tBar, vBar]]), xBar])
    
    # if (tBar, vBar, xBar) belongs to negative dual cone
    if (tBar < 0 and np.all(xBar) < 0 and
         (vBar <= tBar * (-n - np.sum(np.log(xBar/tBar))))):
        return np.zeros((n+2,)), stats
    
    # special case with analytic solution
    if (vBar <= 0 and tBar >= 0):
        return np.concatenate([np.array([[tBar, 0]]), np.maximum(xBar, 0)]), stats
    
    nGradientSteps = 0
    v0 = np.max((vBar, 1))
    x0 = np.maximum(xBar, 1)
    var0 = np.concatenate([np.array([[v0]]), x0])
    var = var0.reshape((n+1, 1))

    for iter in range(1, max_iter + 1):
        v, x = var[0, 0], var[1:]
        x = np.maximum(MIN_X, x)
        
        # if v seems to converge to 0 we return the origin as the solution
        if (v < MIN_V):
            var[0] = 0
            var[1:] = np.maximum(xBar, 0)
            newtonDec = -1337
            newObj = -1337
            break

        assert(v > MIN_FLOAT and np.min(x) > MIN_FLOAT)
        vInv = 1/v
        temp0 = - np.sum(np.log(x*vInv))
        a, c = v * temp0 - tBar, temp0 + n 
        z = 1/x

        # compute gradient
        gradVec = np.zeros((n+1, 1))
        gradVec[0] = a * c + v - vBar
        gradVec[1:] = -a*v*z + x - xBar
        
        # compute Hessian components
        d = np.ones((n+1, 1))
        d[0] += a * (-a / v**2 + n/v - 2*c/v)
        d[1:] += a*v * z**2        
        w = np.zeros((n+1, 1))
        w[0] = -(a + v * c) / v
        w[1:] = v * z 

           
        temp1, temp2 = -gradVec/d, w/d
        assert(abs(1 + w.T @ temp2) > MIN_FLOAT)
        dVar = temp1 - (w.T @ temp1) / (1 + w.T @ temp2) * temp2

        dirDer = gradVec.T @ dVar
        dirDer = dirDer[0, 0]
    
        if (dirDer > 0):
            dVar = -gradVec 
            dirDer = -gradVec.T @ gradVec
            nGradientSteps += 1
       
        newtonDec = -dirDer
    
        if newtonDec <= 2*tol:
            break
        
        idxs = (dVar < 0)
        if np.any(idxs):
            t = np.min((1, 0.99*np.min(-var[idxs] / dVar[idxs])))
        else:
            t = 1

        var_new = var + t*dVar
        objOld = objVal(var, tBar, vBar, xBar, n)
        newObj = objVal(var_new, tBar, vBar, xBar, n)
        
        # newObj will always be positive so the robustness check makes sense
        while (1-LINESEARCH_RELATIVE_TOL)*newObj  > objOld + alpha*t*dirDer: 
            t *= beta 
            var_new = var + t*dVar 
            newObj = objVal(var_new, tBar, vBar, xBar, n)

        # if the absolute difference between two iterates is small we return
        if (np.linalg.norm(var_new - var, np.inf) < TOL_ABSOLUTE_DIFF_INF):
            print("Terminating because of relative difference")
            break

        var = var_new 

    end = time()
    stats = {"obj": newObj, "iter": iter, "newtonDec": newtonDec, 
             "time": end - start, "nGradientSteps": nGradientSteps}

    if (v < MIN_V):
        var = np.concatenate([np.array([[np.maximum(tBar, 0)]]), var])
    else:
        assert(v > MIN_FLOAT and np.min(x) > MIN_FLOAT)
        tOpt = - v * (np.sum(np.log(x)) - n * np.log(v)) 
        var = np.concatenate([np.array([[tOpt]]), var])

    return var, stats 

# ------------------------------------------------------------------
#  Code for projecting onto the nuclear norm cone
# ------------------------------------------------------------------

# projects a nonnegative and sorted vector onto the ell1 norm cone
def projEllOneAlreadyNonNegAndSorted(t0, x0, n):
    # assert that x0 is sorted and positive 
    sumX = 0

    if -t0 >= x0[0]:
        return np.zeros((n+1,))

    for k in range(1, n): 
        sumX += x0[k-1]
        tempSum = (-t0 + sumX)/(k+1)
        if (x0[k-1] > tempSum) and (x0[k] <= tempSum):
            break 
    
    if (x0[k] > tempSum):
        k = n
        sumX += x0[n-1]

    t = -t0 + sumX

    if t > 0:
        t = t0 + t / (k + 1)
    else:
        t = t0
    
    x = np.copy(x0)
    x[0:k] -= (t - t0)
    x[k:] = 0

    return np.concatenate([np.array([t]), x]).reshape((n+1, ))

# projects (t, X) where X is m x n with m >= n onto the nuclear norm cone
def nuclearConeProjPy(t0, X0, m, n):

    # Compute SVD
    U, S, VT = np.linalg.svd(X0, full_matrices=False)

    # Project onto spectral vector cone
    projtS = projEllOneAlreadyNonNegAndSorted(t0, S, n)

    # Recover projection onto spectral matrix cone
    proj = U @ np.diag(projtS[1:]) @ VT

    return projtS[0], proj
