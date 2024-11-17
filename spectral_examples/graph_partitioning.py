import networkx as nx
import numpy as np
import cvxpy as cp
import pdb
from utilsExamples import vectorize
import scipy.sparse as sp
import scs

print("\n \n \n Running graph_partitioning.py \n \n \n")

tol = 1e-4
np.random.seed(0)
def cvxpy_solve(L, n, k):
    # solve with cvxpy
    x = cp.Variable(n)
    objective = cp.Minimize(cp.lambda_sum_largest(-L + cp.diag(x), k))
    constraints = [cp.sum(x) == 0]
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, eps_abs=tol, eps_rel=tol, verbose = True)
    return prob.solution

def create_custom_matrix(n):
    sd_size = int(n * (n + 1) / 2)
    matrix = sp.lil_matrix((sd_size, n + 1), dtype=float)
    idx = 0
    for i in range(n):
        matrix[idx, i + 1] = 1.0
        idx += (n-i)
    
    return matrix

def SpectralSCS_solve(L, n, k):
    sd_size = int(n * (n + 1) / 2)

    c = np.zeros((n+1, ))
    c[0] = 1.0 
    b = np.zeros((2 + sd_size, ))
    b[2:] = -vectorize(L)

    row1 = np.zeros((1, n + 1))
    row1[0, 1:] = np.ones((n, ))
    row2 = np.zeros((1, n + 1))
    row2[0, 0] = -1.0
    row3 = create_custom_matrix(n)
    A = sp.vstack((row1, row2, -row3))
   
    K = {"z": 1, "sl_n": [n], "sl_k": [k]}
    data = {'A': sp.csc_matrix(A), 'b': b, 'c': c}
    solver = scs.SCS(data, K, max_iters=int(1e4), adaptive_scale=True, verbose=True)
    sol = solver.solve()

    return sol

all_n = [100, 200, 300, 400, 500]
num_runs = 5
all_iter = np.zeros((num_runs, len(all_n), 2))
all_solve_times = np.zeros((num_runs, len(all_n), 2))
all_matrix_proj_times = np.zeros((num_runs, len(all_n), 2))
all_vector_proj_times = np.zeros((num_runs, len(all_n)))
all_cone_times = np.zeros((num_runs, len(all_n), 2))
all_lin_sys_times = np.zeros((num_runs, len(all_n), 2))

p = 0.01
k = 10

for i in range(len(all_n)):
    n = all_n[i]
    for run in range(0, num_runs):
        print("-------------------------------")
        print("         n / run: ", n, run)
        print("-------------------------------")
        G = nx.erdos_renyi_graph(n, p)
        L = nx.laplacian_matrix(G).toarray().astype(np.float64)
      
        sol_logdet = SpectralSCS_solve(L, n, k)
        all_solve_times[run, i, 0] = sol_logdet['info']['solve_time']
        all_iter[run, i, 0] = sol_logdet['info']['iter']
        all_matrix_proj_times[run, i, 0] = sol_logdet['info']['ave_time_matrix_cone_proj']
        all_vector_proj_times[run, i] = sol_logdet['info']['ave_time_vector_cone_proj']
        all_cone_times[run, i, 0] = sol_logdet['info']['cone_time'] 
        all_lin_sys_times[run, i, 0] = sol_logdet['info']['lin_sys_time'] 

        sol_cvxpy = cvxpy_solve(L, n, k)
        info_cvxpy = sol_cvxpy.attr['solver_specific_stats']['info']
        all_solve_times[run, i, 1] = info_cvxpy['solve_time']
        all_iter[run, i, 1] = info_cvxpy['iter']
        all_matrix_proj_times[run, i, 1] = info_cvxpy['ave_time_matrix_cone_proj']
        all_cone_times[run, i, 1] = info_cvxpy['cone_time']         
        all_lin_sys_times[run, i, 1] = info_cvxpy['lin_sys_time']                 

np.savez(f'plotting/data/graph_partitioning.npz', 
        all_n=all_n,
        all_iter=all_iter,
        all_solve_times=all_solve_times,
        all_matrix_proj_times=all_matrix_proj_times,
        all_vector_proj_times=all_vector_proj_times,
        all_cone_times=all_cone_times,
        all_lin_sys_times=all_lin_sys_times)