import numpy as np
import seaborn as sns 
import pdb
import matplotlib.pyplot as plt
from SIZES_FIGURES import MARKER_SIZE, FONTSIZE_Y_AXIS, LINEWIDTH, FONTSIZE_X_AXIS, LEGEND_SIZE, TICK_SIZE_X, TICK_SIZE_Y
from matplotlib.ticker import ScalarFormatter

sns.set_theme(style="whitegrid")

def read_data(loaded_data):
    # Access the arrays by their saved names
    all_n = loaded_data['all_n']
    all_iter = loaded_data['all_iter']
    all_solve_times = loaded_data['all_solve_times']
    all_matrix_proj_times = loaded_data['all_matrix_proj_times']
    all_vector_proj_times = loaded_data['all_vector_proj_times']
    all_cone_times = loaded_data['all_cone_times']
    all_lin_sys_times = loaded_data['all_lin_sys_times']

    all_total_times_logdet = all_lin_sys_times[:, :, 0] + all_solve_times[:, :, 0]
    all_total_times_standard = all_lin_sys_times[:, :, 1] + all_solve_times[:, :, 1]
    
    avg_all_iter_logdet = np.mean(all_iter[:, :, 0], axis=0)
    avg_all_solve_times_logdet = np.mean(all_solve_times[:, :, 0], axis=0)
    avg_all_matrix_proj_times_logdet = np.mean(all_matrix_proj_times[:, :, 0], axis=0)
    avg_all_vector_proj_times_logdet = np.mean(all_vector_proj_times, axis=0)
    avg_all_cone_times_logdet = np.mean(all_cone_times[:, :, 0], axis=0)
    avg_all_lin_sys_times_logdet = np.mean(all_lin_sys_times[:, :, 0], axis=0)
    
    avg_all_iter_standard = np.mean(all_iter[:, :, 1], axis=0)
    avg_all_solve_times_standard = np.mean(all_solve_times[:, :, 1], axis=0)
    avg_all_matrix_proj_times_standard = np.mean(all_matrix_proj_times[:, :, 1], axis=0) 
    avg_all_cone_times_standard = np.mean(all_cone_times[:, :, 1], axis=0)
    avg_all_lin_sys_times_standard = np.mean(all_lin_sys_times[:, :, 1], axis=0)

    avg_all_total_time_logdet = (avg_all_solve_times_logdet + avg_all_lin_sys_times_logdet) / 1000
    avg_all_total_time_standard = (avg_all_solve_times_standard + avg_all_lin_sys_times_standard) / 1000

    speedup = avg_all_total_time_standard / avg_all_total_time_logdet
    print("speedup exp design:                           ", speedup)
    print("average speedup exp design:                   ", np.mean(speedup))
    print("speedup computed other way exp design:        ", np.mean(all_total_times_standard / all_total_times_logdet))
    print("\n")

    avg_time_per_iter_logdet =  avg_all_total_time_logdet / avg_all_iter_logdet
    avg_time_per_iter_standard = avg_all_total_time_standard / avg_all_iter_standard

    return (all_n, avg_all_iter_logdet, avg_all_solve_times_logdet,
            avg_all_matrix_proj_times_logdet, avg_all_vector_proj_times_logdet,
            avg_all_cone_times_logdet, avg_all_lin_sys_times_logdet,
            avg_time_per_iter_logdet, avg_all_total_time_logdet, avg_all_iter_standard, 
            avg_all_solve_times_standard, avg_all_matrix_proj_times_standard,
            avg_all_cone_times_standard, avg_all_lin_sys_times_standard,
            avg_time_per_iter_standard, avg_all_total_time_standard)
        


SUBTRACT = 14
SUBTRACT2 = 5

loaded_data = np.load('data/exp_design_data_tol=0.0001.npz')
all_n, avg_all_iter_logdet, avg_all_solve_times_logdet, \
avg_all_matrix_proj_times_logdet, avg_all_vector_proj_times_logdet, \
avg_all_cone_times_logdet, avg_all_lin_sys_times_logdet, \
avg_time_per_iter_logdet, avg_all_total_time_logdet, avg_all_iter_standard,  \
avg_all_solve_times_standard, avg_all_matrix_proj_times_standard, \
avg_all_cone_times_standard, avg_all_lin_sys_times_standard, \
avg_time_per_iter_standard, avg_all_total_time_standard = read_data(loaded_data)

# ----------------------------------------------------------------------------
#                   plot spectral vector cone time
# ----------------------------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.plot(all_n, avg_all_vector_proj_times_logdet, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.plot(all_n, avg_all_matrix_proj_times_logdet, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
plt.subplots_adjust(left=0.20, bottom=0.15, right=0.98, top=0.95)
plt.xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS)
plt.ylabel('time (ms)', fontsize=FONTSIZE_Y_AXIS)
plt.yscale('log')
plt.grid(True)#, which="both", linestyle='--', linewidth=0.5)
plt.savefig(f"figures/exp_design_ablation_high_tol.pdf")


###############################################################################
#                       TOP ROW
###############################################################################
fig, (ax1, ax2) = plt.subplots(2, 4, figsize=(32, 14), sharey=False, sharex=True)
ax1[0].plot(all_n, avg_all_total_time_logdet, label='SpectralSCS', marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[0].plot(all_n, avg_all_total_time_standard, label='SCS', marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[0].set_ylabel("runtime (s)", fontsize=FONTSIZE_Y_AXIS)
ax1[0].grid(True)
ax1[0].set_yscale('log')
ax1[1].plot(all_n, avg_all_iter_logdet, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[1].plot(all_n, avg_all_iter_standard, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[1].grid(True)
ax1[1].set_ylabel("iterations", fontsize=FONTSIZE_Y_AXIS)
ax1[2].plot(all_n, avg_time_per_iter_logdet * 1000, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[2].plot(all_n, avg_time_per_iter_standard * 1000, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[2].grid(True)
ax1[2].set_ylabel("iteration time (ms)", fontsize=FONTSIZE_Y_AXIS)
ax1[3].plot(all_n, avg_all_matrix_proj_times_logdet, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[3].plot(all_n, avg_all_matrix_proj_times_standard, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[3].grid(True)
ax1[3].set_ylabel("matrix cone projection time (ms)", fontsize=FONTSIZE_Y_AXIS - SUBTRACT)


ax1[0].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
ax1[1].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
ax1[2].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
ax1[3].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)

formatter.set_powerlimits((3, 3))
ax1[1].yaxis.set_major_formatter(formatter)
ax1[1].yaxis.get_offset_text().set_fontsize(FONTSIZE_Y_AXIS - SUBTRACT2)
formatter.set_powerlimits((2, 2))
ax1[2].yaxis.set_major_formatter(formatter)
ax1[2].yaxis.get_offset_text().set_fontsize(FONTSIZE_Y_AXIS - SUBTRACT2)
ax1[3].yaxis.set_major_formatter(formatter)
ax1[3].yaxis.get_offset_text().set_fontsize(FONTSIZE_Y_AXIS - SUBTRACT2)

fig.legend(fontsize=LEGEND_SIZE, loc='upper center', bbox_to_anchor=(0.52, 1), ncol=2)



###############################################################################
#                       Bottom row
###############################################################################
loaded_data = np.load('data/exp_design_data_tol=0.001.npz')
all_n, avg_all_iter_logdet, avg_all_solve_times_logdet, \
avg_all_matrix_proj_times_logdet, avg_all_vector_proj_times_logdet, \
avg_all_cone_times_logdet, avg_all_lin_sys_times_logdet, \
avg_time_per_iter_logdet, avg_all_total_time_logdet, avg_all_iter_standard,  \
avg_all_solve_times_standard, avg_all_matrix_proj_times_standard, \
avg_all_cone_times_standard, avg_all_lin_sys_times_standard, \
avg_time_per_iter_standard, avg_all_total_time_standard = read_data(loaded_data)
ax2[0].plot(all_n, avg_all_total_time_logdet, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[0].plot(all_n, avg_all_total_time_standard, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[0].set_ylabel("runtime (s)", fontsize=FONTSIZE_Y_AXIS)
ax2[0].grid(True)
ax2[0].set_yscale('log')
ax2[1].plot(all_n, avg_all_iter_logdet, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[1].plot(all_n, avg_all_iter_standard, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[1].grid(True)
ax2[1].set_ylabel("iterations", fontsize=FONTSIZE_Y_AXIS)
ax2[2].plot(all_n, avg_time_per_iter_logdet * 1000, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[2].plot(all_n, avg_time_per_iter_standard * 1000, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[2].grid(True)
ax2[2].set_ylabel("iteration time (ms)", fontsize=FONTSIZE_Y_AXIS)
ax2[3].plot(all_n, avg_all_matrix_proj_times_logdet, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[3].plot(all_n, avg_all_matrix_proj_times_standard, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[3].grid(True)
ax2[3].set_ylabel("matrix cone projection time (ms)", fontsize=FONTSIZE_Y_AXIS - SUBTRACT)


ax2[0].set_xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS)
ax2[1].set_xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS)
ax2[2].set_xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS)
ax2[3].set_xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS)

ax2[0].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
ax2[1].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
ax2[2].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
ax2[3].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)

formatter.set_powerlimits((3, 3))
ax2[1].yaxis.set_major_formatter(formatter)
ax2[1].yaxis.get_offset_text().set_fontsize(FONTSIZE_Y_AXIS - SUBTRACT2)
formatter.set_powerlimits((2, 2))
ax2[2].yaxis.set_major_formatter(formatter)
ax2[2].yaxis.get_offset_text().set_fontsize(FONTSIZE_Y_AXIS - SUBTRACT2)
ax2[3].yaxis.set_major_formatter(formatter)
ax2[3].yaxis.get_offset_text().set_fontsize(FONTSIZE_Y_AXIS - SUBTRACT2)

plt.subplots_adjust(left=0.05, right=0.98, wspace=0.30, hspace=0.15)

plt.savefig("figures/exp_design_new.pdf")


# ----------------------------------------------------------------------------
#                   plot spectral vector cone time
# ----------------------------------------------------------------------------
plt.figure(figsize=(8, 8))
plt.plot(all_n, avg_all_vector_proj_times_logdet, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.plot(all_n, avg_all_matrix_proj_times_logdet, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
plt.subplots_adjust(left=0.20, bottom=0.15, right=0.98, top=0.95)
plt.xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS)
plt.ylabel('time (ms)', fontsize=FONTSIZE_Y_AXIS)
plt.yscale('log')
plt.grid(True)#, which="both", linestyle='--', linewidth=0.5)
plt.savefig(f"figures/exp_design_ablation_;ow_tol.pdf")
