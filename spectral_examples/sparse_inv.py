import scs
import numpy as np
import scipy.sparse as sp
from utilsExamples import vectorize
import cvxpy as cp

print("\n \n \n Running sparse_inv.py \n \n \n")

# Generates a positive definite matrix with sparse inverse
def generateS(n, density=0.01):
    rvs = lambda s: np.random.choice([-1, 1], size=s) 
    U = sp.random(n, n, density=density, data_rvs=rvs, format='coo')
    A = U.T @ U 
    d = np.diag(A.toarray())
    A = np.maximum(np.minimum(A - np.diag(d), 1), -1)
    B = A + np.diag(1 + d)
    SigmaInv = B + np.max((-1.2 * np.min((1, np.min(np.linalg.eigvalsh(B)))), 0.001)) * np.eye(n)
    Sigma = np.linalg.inv(SigmaInv)
    E = 2 * np.random.rand(n, n) - 1
    E = 0.5 * (E + E.T)
    E = E / np.linalg.norm(E, 'fro')
    S = Sigma + 0.5 * np.linalg.norm(Sigma, 'fro') * E 
    S = S + np.max((np.min(np.linalg.eigvalsh(S)), 0.001)) * np.eye(n)
    return S, SigmaInv

# helper function for canonicalization to logdet cone
def create_custom_diagonal_matrix(n):
    sd = int(n * (n + 1) / 2) 
    diag_values = np.full(sd, np.sqrt(2))
    indices = [0]
    for k in range(1, n):
        indices.append(indices[k - 1] +  1 + n - k)
    diag_values[indices] = 1
    diagonal_matrix = sp.csc_matrix((diag_values, (np.arange(sd), np.arange(sd))), shape=(sd, sd))
    return diagonal_matrix

# solve using SCS with logdet cone
def SpectralSCS_solve(S, n, reg, csv_filename):
    sd_size = int(n * (n + 1) / 2)
    nVariables = 2 * sd_size + 2 

    # Create constraint matrix
    D = create_custom_diagonal_matrix(n)
    row1 = np.zeros((1, nVariables))
    row1[0, 1] = 1.0
    row2 = sp.hstack((sp.csc_matrix((sd_size, 2)), D, -sp.identity(sd_size)))
    row3 = sp.hstack((sp.csc_matrix((sd_size, 2)), -D, -sp.identity(sd_size)))
    row4 = np.zeros((1, nVariables))
    row4[0, 0] = -1.0
    row5 = np.zeros((1, nVariables))
    row5[0, 1] = -1.0
    row6 = sp.hstack((sp.csc_matrix((sd_size, 2)), -sp.identity(sd_size), sp.csc_matrix((sd_size, sd_size))))
    A = sp.vstack((row1, row2, row3, row4, row5, row6))
    
    b = np.zeros((3 + 3 * sd_size, ))
    b[0] = 1
    c = np.hstack([1, 0, vectorize(S), reg * np.ones((sd_size, ))])
    K = {"z": 1, "l": 2*sd_size, "q": [], "s": [], "ep": 0, "ed": 0,
        "p": [], "d": [n]}
    data = {'A': sp.csc_matrix(A), 'b': b, 'c': c}

    if csv_filename == None:
        solver = scs.SCS(data, K, adaptive_scale=False,  eps_abs=1e-4, eps_rel=1e-4)
    else:
        solver = scs.SCS(data, K, adaptive_scale=False,  eps_abs=1e-4, eps_rel=1e-4, 
                         log_csv_filename=csv_filename)
    sol = solver.solve()
    return sol 

    # solves standard canonicalization using SCS
def SCS_solve(S, n, reg):
    X = cp.Variable((n, n), symmetric=True)
    objective = cp.Minimize(cp.trace(S @ X) - cp.log_det(X) + reg * cp.sum(cp.abs(X)))
    problem = cp.Problem(objective)
    problem.solve(verbose=True, solver=cp.SCS, max_iters=int(1e4), eps_abs=1e-4, eps_rel=1e-4)
    return problem.solution

all_rho =    [0.1,  0.075,  0.06,  0.045,  0.045,   0.038]
all_n =      [50,     100,   150,    200,   250,     300]
densities =  [0.03, 0.020, 0.020,  0.018, 0.016,   0.014]
num_runs = 5
all_iter = np.zeros((num_runs, len(all_n), 2))
all_solve_times = np.zeros((num_runs, len(all_n), 2))
all_matrix_proj_times = np.zeros((num_runs, len(all_n), 2))
all_vector_proj_times = np.zeros((num_runs, len(all_n)))
all_cone_times = np.zeros((num_runs, len(all_n), 2))
all_lin_sys_times = np.zeros((num_runs, len(all_n), 2))
solve_with_SpectralSCS = True 
solve_with_SCS = True 
log_to_csv = False

# solve with spectral SCS
seed = 0
np.random.seed(seed)
if solve_with_SpectralSCS:
    for i in range(len(all_n)):
        n = all_n[i]
        for run in range(num_runs):
            print("---------------------------------")
            print("  SpectralSCS  n / run: ", n, run)
            print("---------------------------------")
            S, _ = generateS(n, density=densities[i])

            if log_to_csv:
                csv_filename = f"csv/sparse_inv/sparse_inv_n={n}_run={run}.csv"
            else:
                csv_filename = None
            sol_logdet = SpectralSCS_solve(S, n, all_rho[i], csv_filename=csv_filename)
            all_solve_times[run, i, 0] = sol_logdet['info']['solve_time']
            all_iter[run, i, 0] = sol_logdet['info']['iter']
            all_matrix_proj_times[run, i, 0] = sol_logdet['info']['ave_time_matrix_cone_proj']
            all_vector_proj_times[run, i] = sol_logdet['info']['ave_time_vector_cone_proj']
            all_cone_times[run, i, 0] = sol_logdet['info']['cone_time'] 
            all_lin_sys_times[run, i, 0] = sol_logdet['info']['lin_sys_time'] 


# solve with standard SCS (resetting the seed guarantees that SCS solves the
# exact same problem instances as SpectralSCS)  
np.random.seed(seed)
if solve_with_SCS:  
    for i in range(len(all_n)):
        n = all_n[i]
        for run in range(num_runs):
            print("---------------------------------")
            print("  SCS  n / run: ", n, run)
            print("---------------------------------")
            S, _ = generateS(n, density=densities[i])      
            sol_cvxpy = SCS_solve(S, n, all_rho[i])
            info_cvxpy = sol_cvxpy.attr['solver_specific_stats']['info']
            all_solve_times[run, i, 1] = info_cvxpy['solve_time']
            all_iter[run, i, 1] = info_cvxpy['iter']
            all_matrix_proj_times[run, i, 1] = info_cvxpy['ave_time_matrix_cone_proj']
            all_cone_times[run, i, 1] = info_cvxpy['cone_time']         
            all_lin_sys_times[run, i, 1] = info_cvxpy['lin_sys_time']

np.savez(f'plotting/data/sparse_inv.npz', 
        all_n=all_n,
        all_iter=all_iter,
        all_solve_times=all_solve_times,
        all_matrix_proj_times=all_matrix_proj_times,
        all_vector_proj_times=all_vector_proj_times,
        all_cone_times=all_cone_times,
        all_lin_sys_times=all_lin_sys_times)