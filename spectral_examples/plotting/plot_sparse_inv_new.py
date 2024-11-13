import numpy as np
import seaborn as sns 
import pdb
import matplotlib.pyplot as plt
from SIZES_FIGURES import MARKER_SIZE, FONTSIZE_Y_AXIS, LINEWIDTH, FONTSIZE_X_AXIS, LEGEND_SIZE, TICK_SIZE_X, TICK_SIZE_Y
from matplotlib.ticker import ScalarFormatter, LogFormatter

sns.set_theme(style="whitegrid")
loaded_data = np.load('sparse_inv_run_data.npz')

# Access the arrays by their saved names
all_n = loaded_data['all_n']
all_iter_logdet = loaded_data['all_iter_logdet']
all_solve_times_logdet = loaded_data['all_solve_times_logdet']
all_matrix_proj_times_logdet = loaded_data['all_matrix_proj_times_logdet']
all_vector_proj_times_logdet = loaded_data['all_vector_proj_times_logdet']
all_cone_times_logdet = loaded_data['all_cone_times_logdet']
all_lin_sys_times_logdet = loaded_data['all_lin_sys_times_logdet']
all_iter_standard = loaded_data['all_iter_standard']
all_solve_times_standard = loaded_data['all_solve_times_standard']
all_matrix_proj_times_standard = loaded_data['all_matrix_proj_times_standard']
all_cone_times_standard = loaded_data['all_cone_times_standard']
all_lin_sys_times_standard = loaded_data['all_lin_sys_times_standard']

avg_all_iter_logdet = np.mean(all_iter_logdet, axis=0)
avg_all_solve_times_logdet = np.mean(all_solve_times_logdet, axis=0)
avg_all_matrix_proj_times_logdet = np.mean(all_matrix_proj_times_logdet, axis=0)
avg_all_vector_proj_times_logdet = np.mean(all_vector_proj_times_logdet, axis=0)
avg_all_cone_times_logdet = np.mean(all_cone_times_logdet, axis=0)
avg_all_lin_sys_times_logdet = np.mean(all_lin_sys_times_logdet, axis=0)
avg_all_iter_standard = np.mean(all_iter_standard, axis=0)
avg_all_solve_times_standard = np.mean(all_solve_times_standard, axis=0)
avg_all_matrix_proj_times_standard = np.mean(all_matrix_proj_times_standard, axis=0)
avg_all_cone_times_standard = np.mean(all_cone_times_standard, axis=0)
avg_all_lin_sys_times_standard = np.mean(all_lin_sys_times_standard, axis=0)

avg_all_total_time_logdet = (avg_all_solve_times_logdet + avg_all_lin_sys_times_logdet) / 1000
avg_all_total_time_standard = (avg_all_solve_times_standard + avg_all_lin_sys_times_standard) / 1000

avg_all_iter_time_logdet =  avg_all_total_time_logdet / avg_all_iter_logdet
avg_all_iter_time_standard = avg_all_total_time_standard / avg_all_iter_standard


SUBTRACT = 14
SUBTRACT2 = 5

# --------------------------------------------------------------------------
#                             plot stats
# --------------------------------------------------------------------------
fig, axs = plt.subplots(1, 4, figsize=(32, 8.6), sharey=False)
axs[0].plot(all_n, avg_all_total_time_logdet, label='SpectralSCS', marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
axs[0].plot(all_n, avg_all_total_time_standard, label='SCS', marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
axs[0].set_ylabel("runtime (s)", fontsize=FONTSIZE_Y_AXIS)
axs[0].grid(True)
#axs[0].legend(fontsize=12)
axs[1].plot(all_n, avg_all_iter_logdet, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
axs[1].plot(all_n, avg_all_iter_standard, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
axs[1].grid(True)
axs[1].set_ylabel("iterations", fontsize=FONTSIZE_Y_AXIS)
axs[2].plot(all_n, avg_all_iter_time_logdet * 1000, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
axs[2].plot(all_n, avg_all_iter_time_standard * 1000, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
axs[2].grid(True)
axs[2].set_ylabel("iteration time (ms)", fontsize=FONTSIZE_Y_AXIS)
axs[3].plot(all_n, avg_all_matrix_proj_times_logdet, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
axs[3].plot(all_n, avg_all_matrix_proj_times_standard, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
axs[3].grid(True)
axs[3].set_ylabel("matrix cone projection time (ms)", fontsize=FONTSIZE_Y_AXIS - SUBTRACT)

axs[0].set_xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS)
axs[1].set_xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS)
axs[2].set_xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS)
axs[3].set_xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS)

axs[0].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
axs[1].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
axs[2].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
axs[3].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)



formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)

formatter.set_powerlimits((3, 3))
axs[1].yaxis.set_major_formatter(formatter)
axs[1].yaxis.get_offset_text().set_fontsize(FONTSIZE_Y_AXIS - SUBTRACT2)
formatter.set_powerlimits((2, 2))
axs[2].yaxis.set_major_formatter(formatter)
axs[2].yaxis.get_offset_text().set_fontsize(FONTSIZE_Y_AXIS - SUBTRACT2)
axs[3].yaxis.set_major_formatter(formatter)
axs[3].yaxis.get_offset_text().set_fontsize(FONTSIZE_Y_AXIS - SUBTRACT2)

fig.legend(fontsize=LEGEND_SIZE, loc='upper center', bbox_to_anchor=(0.52, 1.01), ncol=2)

speedup = avg_all_total_time_standard / avg_all_total_time_logdet
print("speed up factor: ", speedup)
print("average speed up: ", np.mean(speedup))
plt.subplots_adjust(left=0.05, right=0.98, top=0.83, wspace=0.3)
plt.savefig("sparse_inv_new.pdf")

# ----------------------------------------------------------------------------
#                   plot spectral vector cone time
# ----------------------------------------------------------------------------
if True:
    plt.figure(figsize=(8, 8))
    plt.plot(all_n, avg_all_vector_proj_times_logdet, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
    plt.plot(all_n, avg_all_matrix_proj_times_logdet, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
    plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
    plt.subplots_adjust(left=0.20, bottom=0.15, right=0.98, top=0.95)
    plt.xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS)
    plt.ylabel('time (ms)', fontsize=FONTSIZE_Y_AXIS)
    plt.yscale('log')
    plt.grid(True)#, which="both", linestyle='--', linewidth=0.5)
    plt.savefig(f"figures/sparse_inv_ablation.pdf")