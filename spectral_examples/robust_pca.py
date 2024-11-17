import scs
import numpy as np
import scipy.sparse as sp
import pdb
import cvxpy as cp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("ratio", type=int)
args = parser.parse_args()
ratio = args.ratio

print(f"\n \n \n robust_pca for ratio {ratio} \n \n \n")

def SpectralSCS_solve(M, mu, csv_filename):
    m, n = M.shape 
    nVariables = 3*m*n + 1

    # Create constraint matrix
    row1 = sp.hstack((sp.csc_matrix((m*n, m*n)), sp.identity(m*n), sp.csc_matrix((m*n, 1)), sp.identity(m*n)))
    row2 = np.zeros((1, nVariables))
    row2[0, 0:m*n] = np.ones((m*n, ))
    row3 = sp.hstack((-sp.identity(m*n), sp.identity(m*n), sp.csc_matrix((m*n, m*n+1))))
    row4 = sp.hstack((-sp.identity(m*n), -sp.identity(m*n), sp.csc_matrix((m*n, m*n+1))))
    row5 = np.zeros((1, nVariables))
    row5[0, 2*m*n] = -1.0
    row6 = sp.hstack((sp.csc_matrix((m*n, 2*m*n+1)), -sp.identity(m*n))) 
    A = sp.vstack((row1, row2, row3, row4, row5, row6))

    b = np.zeros((4*m*n + 2, ))
    b[0:m*n] = M.flatten(order='F')
    b[m*n] = mu 

    c = np.zeros((nVariables, ))
    c[2*m*n] = 1.0
    K = {"z": m*n, "l": 2*m*n + 1, "q": [], "s": [], "ep": 0, "ed": 0,
        "p": [], "d": [], "nuc_m": [m], "nuc_n": [n]}
    
    data = {'A': sp.csc_matrix(A), 'b': b, 'c': c}
    if csv_filename == None:
        solver = scs.SCS(data, K, max_iters=int(1e4), eps_abs=1e-4, eps_rel=1e-4)
    else:
        solver = scs.SCS(data, K, max_iters=int(1e4), eps_abs=1e-4, eps_rel=1e-4, 
                         log_csv_filename=csv_filename)
    sol = solver.solve()
    return sol

def SCS_solve(M, mu):
    m, n = M.shape 
    X = cp.Variable((m, n))
    S = cp.Variable((m, n))
    obj = cp.Minimize(cp.norm(X, "nuc"))
    constrs = [X + S == M, cp.sum(cp.abs(S)) <= mu]
    prob = cp.Problem(objective=obj, constraints=constrs)
    prob.solve(verbose=True, solver=cp.SCS, eps_abs=1e-4, eps_rel=1e-4, max_iters=int(1e4))
    return prob.solution

def generate_data(m, n, rank):
    L_hat = np.random.rand(m, rank) @ np.random.rand(rank, n)
    S_hat = 0.1*np.max(np.abs(L_hat))*sp.random(m, n, density= 0.1, data_rvs=np.random.randn)
    M = L_hat + S_hat 
    mu = 1 * np.sum(np.abs(S_hat))
    return M, mu

all_m =  [100, 150, 200, 250, 300]
rank = 10
num_runs = 5
all_iter = np.zeros((num_runs, len(all_m), 2))
all_solve_times = np.zeros((num_runs, len(all_m), 2))
all_matrix_proj_times = np.zeros((num_runs, len(all_m), 2))
all_vector_proj_times = np.zeros((num_runs, len(all_m)))
all_cone_times = np.zeros((num_runs, len(all_m), 2))
all_lin_sys_times = np.zeros((num_runs, len(all_m), 2))
solve_with_SpectralSCS = True 
solve_with_SCS = True 
log_to_csv = False

# solve with spectral SCS
seed = 0
np.random.seed(seed)
if solve_with_SpectralSCS:
    for i in range(len(all_m)):
        m = all_m[i]
        n = int(m / ratio)
        for run in range(num_runs):
            print("---------------------------------")
            print("  SpectralSCS  n / run: ", n, run)
            print("---------------------------------")
            M, mu = generate_data(m, n, rank)

            if log_to_csv:
                csv_filename = f"csv/robust_pca/pca_m={m}_run={run}.csv"
            else:
                csv_filename = None
            sol_logdet = SpectralSCS_solve(M, mu, csv_filename=csv_filename)
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
    for i in range(len(all_m)):
        m = all_m[i]
        n = int(m / ratio)
        for run in range(num_runs):
            print("---------------------------------")
            print("  SCS  n / run: ", n, run)
            print("---------------------------------")
            M, mu = generate_data(m, n, rank)      
            sol_cvxpy = SCS_solve(M, mu)
            info_cvxpy = sol_cvxpy.attr['solver_specific_stats']['info']
            all_solve_times[run, i, 1] = info_cvxpy['solve_time']
            all_iter[run, i, 1] = info_cvxpy['iter']
            all_matrix_proj_times[run, i, 1] = info_cvxpy['ave_time_matrix_cone_proj']
            all_cone_times[run, i, 1] = info_cvxpy['cone_time']         
            all_lin_sys_times[run, i, 1] = info_cvxpy['lin_sys_time']

        
np.savez(f'plotting/data/robust_pca_ratio={ratio}.npz', 
        all_m=all_m,
        all_iter=all_iter,
        all_solve_times=all_solve_times,
        all_matrix_proj_times=all_matrix_proj_times,
        all_vector_proj_times=all_vector_proj_times,
        all_cone_times=all_cone_times,
        all_lin_sys_times=all_lin_sys_times)