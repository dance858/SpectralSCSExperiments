import scs
import numpy as np
import scipy.sparse as sp
from utilsExamples import vectorize
import cvxpy as cp


print("\n \n \n Running exp_design.py \n \n \n")

tol = 1e-4

def SpectralSCS_solve(F, csv_filename):
    n = F.shape[0]
    p = F.shape[1]
    sd_size = int(n * (n + 1) / 2)
    nVariables = 2 + sd_size 

    row1 = np.zeros((1, nVariables))
    row1[0, 1] = 1.0 

    row2 = np.zeros((p, nVariables))
    for i in range(0, p):
        row2[i, 2:] = vectorize(np.outer(F[:, i], F[:, i]))
    
    row3 = -sp.identity(sd_size + 2)

    A = sp.vstack((row1, row2, row3))
    b = np.zeros((3 + p + sd_size,))
    b[0:1+p] = 1.0 
    c = np.hstack([1, 0, np.zeros((sd_size, ))])
    K = {"z": 1, "l": p, "d":[n]}

    data = {'A': sp.csc_matrix(A), 'b': b, 'c': c}
    if csv_filename == None:
        solver = scs.SCS(data, K, adaptive_scale=False, eps_abs=tol, eps_rel=tol,
                     max_iters=int(1e4), verbose=True)
    else:
        solver = scs.SCS(data, K, adaptive_scale=False, eps_abs=tol, eps_rel=tol,
                     max_iters=int(1e4), verbose=True, log_csv_filename=csv_filename)
    sol = solver.solve()
    return sol

 
def SCS_solve(F):
    n = F.shape[0]
    p = F.shape[1]
    W = cp.Variable((n, n), symmetric=True)
    objective = cp.Minimize(-cp.log_det(W))
    constraints = []
    for i in range(0, p):
        constraints.append(cp.quad_form(F[:, i], W) <= 1)
    problem = cp.Problem(objective, constraints)
    problem.solve(verbose=True, solver=cp.SCS, eps_abs=tol, eps_rel=tol, max_iters=int(1e4))
    return problem.solution

all_n = [50, 100, 150, 200, 250, 300]
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
        p = 2*n 
        for run in range(num_runs):
            print("---------------------------------")
            print(f"  SpectralSCS:  n={n}, run={run} ")
            print("---------------------------------")
            F = np.random.randn(n, p)

            if log_to_csv:
                csv_filename = f"csv/exp_design_new/exp_des_n={n}_run={run}_tol={tol}.csv"
            else:
                csv_filename = None
            sol_logdet = SpectralSCS_solve(F, csv_filename=csv_filename)
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
        p = 2*n 
        for run in range(num_runs):
            print("---------------------------------")
            print("  SCS  n / run: ", n, run)
            print("---------------------------------")
            F = np.random.randn(n, p)       
            sol_cvxpy = SCS_solve(F)
            info_cvxpy = sol_cvxpy.attr['solver_specific_stats']['info']
            all_solve_times[run, i, 1] = info_cvxpy['solve_time']
            all_iter[run, i, 1] = info_cvxpy['iter']
            all_matrix_proj_times[run, i, 1] = info_cvxpy['ave_time_matrix_cone_proj']
            all_cone_times[run, i, 1] = info_cvxpy['cone_time']         
            all_lin_sys_times[run, i, 1] = info_cvxpy['lin_sys_time']     

np.savez(f'plotting/data/exp_design_data_tol={tol}.npz', 
        all_n=all_n,
        all_iter=all_iter,
        all_solve_times=all_solve_times,
        all_matrix_proj_times=all_matrix_proj_times,
        all_vector_proj_times=all_vector_proj_times,
        all_cone_times=all_cone_times,
        all_lin_sys_times=all_lin_sys_times)