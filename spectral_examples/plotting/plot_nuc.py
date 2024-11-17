import numpy as np
import seaborn as sns 
import pdb
import matplotlib.pyplot as plt
from SIZES_FIGURES import MARKER_SIZE, FONTSIZE_Y_AXIS, LINEWIDTH, FONTSIZE_X_AXIS, LEGEND_SIZE, TICK_SIZE_X, TICK_SIZE_Y
from matplotlib.ticker import ScalarFormatter, LogFormatter

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------
#                        For ratio = 1
# --------------------------------------------------------------------------
loaded_data = np.load('data/robust_pca_ratio=1.npz')
all_m_r1 = loaded_data['all_m']
all_iter = loaded_data['all_iter']
all_solve_times = loaded_data['all_solve_times']
all_matrix_proj_times = loaded_data['all_matrix_proj_times']
all_vector_proj_times = loaded_data['all_vector_proj_times']
all_cone_times = loaded_data['all_cone_times']
all_lin_sys_times = loaded_data['all_lin_sys_times']
all_iter_nuc = all_iter[:, :, 0]
all_solve_times_nuc = all_solve_times[:, :, 0]
all_matrix_proj_times_nuc = all_matrix_proj_times[:, :, 0]
all_vector_proj_times_nuc = all_vector_proj_times
all_cone_times_nuc = all_cone_times[:, :, 0]
all_lin_sys_times_nuc =  all_lin_sys_times[:, :, 0]
all_iter_standard = all_iter[:, :, 1]
all_solve_times_standard = all_solve_times[:, :, 1]
all_matrix_proj_times_standard = all_matrix_proj_times[:, :, 1]
all_cone_times_standard = all_cone_times[:, :, 1]
all_lin_sys_times_standard = all_lin_sys_times[:, :, 1]

all_total_times_standard = all_lin_sys_times_standard + all_solve_times_standard
all_total_times_nuc = all_lin_sys_times_nuc + all_solve_times_nuc

avg_iter_nuc_r1 = np.mean(all_iter_nuc, axis=0)
avg_solve_times_nuc_r1 = np.mean(all_solve_times_nuc, axis=0)
avg_matrix_proj_times_nuc_r1 = np.mean(all_matrix_proj_times_nuc, axis=0)
avg_vector_proj_times_nuc_r1 = np.mean(all_vector_proj_times_nuc, axis=0)
avg_cone_times_nuc_r1 = np.mean(all_cone_times_nuc, axis=0)
avg_lin_sys_times_nuc_r1 = np.mean(all_lin_sys_times_nuc, axis=0)

avg_iter_standard_r1 = np.mean(all_iter_standard, axis=0)
avg_solve_times_standard_r1 = np.mean(all_solve_times_standard, axis=0)
avg_matrix_proj_times_standard_r1 = np.mean(all_matrix_proj_times_standard, axis=0)
avg_cone_times_standard_r1 = np.mean(all_cone_times_standard, axis=0)
avg_lin_sys_times_standard_r1 = np.mean(all_lin_sys_times_standard, axis=0)

avg_total_time_nuc_r1 = (avg_solve_times_nuc_r1 + avg_lin_sys_times_nuc_r1) / 1000
avg_total_time_standard_r1 = (avg_solve_times_standard_r1 + avg_lin_sys_times_standard_r1) / 1000
avg_iter_time_nuc_r1 =  avg_total_time_nuc_r1 / avg_iter_nuc_r1
avg_iter_time_standard_r1 = avg_total_time_standard_r1 / avg_iter_standard_r1

speedup =  avg_total_time_standard_r1 / avg_total_time_nuc_r1
print("speedup nuc ratio = 1:         ", speedup)
print("average speedup nuc ratio = 1: ", np.mean(speedup))
print("speedup computed other way:    ", np.mean(all_total_times_standard / all_total_times_nuc), "\n")

# --------------------------------------------------------------------------
#                        For ratio = 2
# --------------------------------------------------------------------------
loaded_data = np.load('data/robust_pca_ratio=2.npz')
all_m_r2 = loaded_data['all_m']
all_iter = loaded_data['all_iter']
all_solve_times = loaded_data['all_solve_times']
all_matrix_proj_times = loaded_data['all_matrix_proj_times']
all_vector_proj_times = loaded_data['all_vector_proj_times']
all_cone_times = loaded_data['all_cone_times']
all_lin_sys_times = loaded_data['all_lin_sys_times']
all_iter_nuc = all_iter[:, :, 0]
all_solve_times_nuc = all_solve_times[:, :, 0]
all_matrix_proj_times_nuc = all_matrix_proj_times[:, :, 0]
all_vector_proj_times_nuc = all_vector_proj_times
all_cone_times_nuc = all_cone_times[:, :, 0]
all_lin_sys_times_nuc =  all_lin_sys_times[:, :, 0]
all_iter_standard = all_iter[:, :, 1]
all_solve_times_standard = all_solve_times[:, :, 1]
all_matrix_proj_times_standard = all_matrix_proj_times[:, :, 1]
all_cone_times_standard = all_cone_times[:, :, 1]
all_lin_sys_times_standard = all_lin_sys_times[:, :, 1]
all_total_times_standard = all_lin_sys_times_standard + all_solve_times_standard
all_total_times_nuc = all_lin_sys_times_nuc + all_solve_times_nuc

avg_iter_nuc_r2 = np.mean(all_iter_nuc, axis=0)
avg_solve_times_nuc_r2 = np.mean(all_solve_times_nuc, axis=0)
avg_matrix_proj_times_nuc_r2 = np.mean(all_matrix_proj_times_nuc, axis=0)
avg_vector_proj_times_nuc_r2 = np.mean(all_vector_proj_times_nuc, axis=0)
avg_cone_times_nuc_r2 = np.mean(all_cone_times_nuc, axis=0)
avg_lin_sys_times_nuc_r2 = np.mean(all_lin_sys_times_nuc, axis=0)

avg_iter_standard_r2 = np.mean(all_iter_standard, axis=0)
avg_solve_times_standard_r2 = np.mean(all_solve_times_standard, axis=0)
avg_matrix_proj_times_standard_r2 = np.mean(all_matrix_proj_times_standard, axis=0)
avg_cone_times_standard_r2 = np.mean(all_cone_times_standard, axis=0)
avg_lin_sys_times_standard_r2 = np.mean(all_lin_sys_times_standard, axis=0)

avg_total_time_nuc_r2 = (avg_solve_times_nuc_r2 + avg_lin_sys_times_nuc_r2) / 1000
avg_total_time_standard_r2 = (avg_solve_times_standard_r2 + avg_lin_sys_times_standard_r2) / 1000
avg_iter_time_nuc_r2 =  avg_total_time_nuc_r2 / avg_iter_nuc_r2
avg_iter_time_standard_r2 = avg_total_time_standard_r2 / avg_iter_standard_r2

speedup =  avg_total_time_standard_r2 / avg_total_time_nuc_r2
print("speedup nuc ratio = 2:         ", speedup)
print("average speedup nuc ratio = 2: ", np.mean(speedup))
print("speedup computed other way:    ", np.mean(all_total_times_standard / all_total_times_nuc), "\n")

# --------------------------------------------------------------------------
#                        For ratio = 5
# --------------------------------------------------------------------------
loaded_data = np.load('data/robust_pca_ratio=5.npz')
all_m_r5 = loaded_data['all_m']
all_iter = loaded_data['all_iter']
all_solve_times = loaded_data['all_solve_times']
all_matrix_proj_times = loaded_data['all_matrix_proj_times']
all_vector_proj_times = loaded_data['all_vector_proj_times']
all_cone_times = loaded_data['all_cone_times']
all_lin_sys_times = loaded_data['all_lin_sys_times']
all_iter_nuc = all_iter[:, :, 0]
all_solve_times_nuc = all_solve_times[:, :, 0]
all_matrix_proj_times_nuc = all_matrix_proj_times[:, :, 0]
all_vector_proj_times_nuc = all_vector_proj_times
all_cone_times_nuc = all_cone_times[:, :, 0]
all_lin_sys_times_nuc =  all_lin_sys_times[:, :, 0]
all_iter_standard = all_iter[:, :, 1]
all_solve_times_standard = all_solve_times[:, :, 1]
all_matrix_proj_times_standard = all_matrix_proj_times[:, :, 1]
all_cone_times_standard = all_cone_times[:, :, 1]
all_lin_sys_times_standard = all_lin_sys_times[:, :, 1]
all_total_times_standard = all_lin_sys_times_standard + all_solve_times_standard
all_total_times_nuc = all_lin_sys_times_nuc + all_solve_times_nuc

avg_iter_nuc_r5 = np.mean(all_iter_nuc, axis=0)
avg_solve_times_nuc_r5 = np.mean(all_solve_times_nuc, axis=0)
avg_matrix_proj_times_nuc_r5 = np.mean(all_matrix_proj_times_nuc, axis=0)
avg_vector_proj_times_nuc_r5 = np.mean(all_vector_proj_times_nuc, axis=0)
avg_cone_times_nuc_r5 = np.mean(all_cone_times_nuc, axis=0)
avg_lin_sys_times_nuc_r5 = np.mean(all_lin_sys_times_nuc, axis=0)

avg_iter_standard_r5 = np.mean(all_iter_standard, axis=0)
avg_solve_times_standard_r5 = np.mean(all_solve_times_standard, axis=0)
avg_matrix_proj_times_standard_r5 = np.mean(all_matrix_proj_times_standard, axis=0)
avg_cone_times_standard_r5 = np.mean(all_cone_times_standard, axis=0)
avg_lin_sys_times_standard_r5 = np.mean(all_lin_sys_times_standard, axis=0)

avg_total_time_nuc_r5 = (avg_solve_times_nuc_r5 + avg_lin_sys_times_nuc_r5) / 1000
avg_total_time_standard_r5 = (avg_solve_times_standard_r5 + avg_lin_sys_times_standard_r5) / 1000
avg_iter_time_nuc_r5 =  avg_total_time_nuc_r5 / avg_iter_nuc_r5
avg_iter_time_standard_r5 = avg_total_time_standard_r5 / avg_iter_standard_r5

speedup =  avg_total_time_standard_r5 / avg_total_time_nuc_r5
print("speedup nuc ratio = 5:         ", speedup)
print("average speedup nuc ratio = 5: ", np.mean(speedup))
print("speedup computed other way:    ", np.mean(all_total_times_standard / all_total_times_nuc), "\n")


SUBTRACT = 14
SUBTRACT2 = 5


# ------------------------------------------------------------------------------
#                               For ratio = 1
# ------------------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 4, figsize=(32, 25), sharey=False, sharex=True)
ax1[0].plot(all_m_r1, avg_total_time_nuc_r1, label='SpectralSCS', marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[0].plot(all_m_r1, avg_total_time_standard_r1, label='SCS', marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[0].set_ylabel("runtime (s)", fontsize=FONTSIZE_Y_AXIS - 13)
ax1[0].set_yscale('log')
ax1[0].grid(True)

ax1[1].plot(all_m_r1, avg_iter_nuc_r1, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[1].plot(all_m_r1, avg_iter_standard_r1, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[1].grid(True)
ax1[1].set_ylabel("iterations", fontsize=FONTSIZE_Y_AXIS - 13)

ax1[2].plot(all_m_r1, avg_iter_time_nuc_r1, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[2].plot(all_m_r1, avg_iter_time_standard_r1, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[2].grid(True)
ax1[2].set_ylabel("iteration time (ms)", fontsize=FONTSIZE_Y_AXIS - 13)

ax1[3].plot(all_m_r1, avg_matrix_proj_times_nuc_r1, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[3].plot(all_m_r1, avg_matrix_proj_times_standard_r1, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[3].grid(True)
ax1[3].set_ylabel("matrix cone projection time (ms)", fontsize=FONTSIZE_X_AXIS - 28)

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


# ------------------------------------------------------------------------------
#                               For ratio = 2
# ------------------------------------------------------------------------------
ax2[0].plot(all_m_r2, avg_total_time_nuc_r2, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[0].plot(all_m_r2, avg_total_time_standard_r2, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[0].set_ylabel("runtime (s)", fontsize=FONTSIZE_Y_AXIS - 13)
ax2[0].set_yscale('log')
ax2[0].grid(True)

ax2[1].plot(all_m_r2, avg_iter_nuc_r2, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[1].plot(all_m_r2, avg_iter_standard_r2, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[1].grid(True)
ax2[1].set_ylabel("iterations", fontsize=FONTSIZE_Y_AXIS - 13)

ax2[2].plot(all_m_r2, avg_iter_time_nuc_r2, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[2].plot(all_m_r2, avg_iter_time_standard_r2, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[2].grid(True)
ax2[2].set_ylabel("iteration time (ms)", fontsize=FONTSIZE_Y_AXIS - 13)

ax2[3].plot(all_m_r2, avg_matrix_proj_times_nuc_r2, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[3].plot(all_m_r2, avg_matrix_proj_times_standard_r2, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[3].grid(True)
ax2[3].set_ylabel("matrix cone projection time (ms)", fontsize=FONTSIZE_X_AXIS - 28)

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

# ------------------------------------------------------------------------------
#                               For ratio = 5
# -----------------------------------------------------------------------------
ax3[0].plot(all_m_r5, avg_total_time_nuc_r5, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[0].plot(all_m_r5, avg_total_time_standard_r5, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[0].set_ylabel("runtime (s)", fontsize=FONTSIZE_Y_AXIS - 13)
ax3[0].set_yscale('log')
ax3[0].grid(True)

ax3[1].plot(all_m_r5, avg_iter_nuc_r5, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[1].plot(all_m_r5, avg_iter_standard_r5, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[1].grid(True)
ax3[1].set_ylabel("iterations", fontsize=FONTSIZE_Y_AXIS - 13)

ax3[2].plot(all_m_r5, avg_iter_time_nuc_r5, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[2].plot(all_m_r5, avg_iter_time_standard_r5, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[2].grid(True)
ax3[2].set_ylabel("iteration time (ms)", fontsize=FONTSIZE_Y_AXIS - 13)

ax3[3].plot(all_m_r5, avg_matrix_proj_times_nuc_r5, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[3].plot(all_m_r5, avg_matrix_proj_times_standard_r5, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[3].grid(True)
ax3[3].set_ylabel("matrix cone projection time (ms)", fontsize=FONTSIZE_X_AXIS - 28)

ax3[0].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
ax3[1].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
ax3[2].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
ax3[3].tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)

formatter = ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((3, 3))
ax3[1].yaxis.set_major_formatter(formatter)
ax3[1].yaxis.get_offset_text().set_fontsize(FONTSIZE_Y_AXIS - SUBTRACT2)
formatter.set_powerlimits((2, 2))
ax3[2].yaxis.set_major_formatter(formatter)
ax3[2].yaxis.get_offset_text().set_fontsize(FONTSIZE_Y_AXIS - SUBTRACT2)
ax3[3].yaxis.set_major_formatter(formatter)
ax3[3].yaxis.get_offset_text().set_fontsize(FONTSIZE_Y_AXIS - SUBTRACT2)


ax3[0].set_xlabel(r'$m$', fontsize=FONTSIZE_X_AXIS)
ax3[1].set_xlabel(r'$m$', fontsize=FONTSIZE_X_AXIS)
ax3[2].set_xlabel(r'$m$', fontsize=FONTSIZE_X_AXIS)
ax3[3].set_xlabel(r'$m$', fontsize=FONTSIZE_X_AXIS)


plt.subplots_adjust(left=0.05, right=0.98, wspace=0.25, hspace=0.18)
fig.legend(fontsize=LEGEND_SIZE, loc='upper center', bbox_to_anchor=(0.52, 0.96), ncol=2)

plt.savefig("figures/robust_pca.pdf") 

# -------------------------------------
# Ablation
# ------------------------------------
subtract = 3
plt.figure(figsize=(8, 8))
plt.plot(all_m_r1, avg_vector_proj_times_nuc_r1, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.plot(all_m_r1, avg_matrix_proj_times_nuc_r1, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
plt.subplots_adjust(left=0.20, bottom=0.15, right=0.98, top=0.95)
plt.xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS - subtract)
plt.ylabel('time (ms)', fontsize=FONTSIZE_Y_AXIS - subtract - 4)
plt.yscale('log')
plt.grid(True)#, which="both", linestyle='--', linewidth=0.5)
plt.savefig(f"figures/nuc_ablation_ratio=1.pdf")

plt.figure(figsize=(8, 8))
plt.plot(all_m_r2, avg_vector_proj_times_nuc_r2, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.plot(all_m_r2, avg_matrix_proj_times_nuc_r2, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
plt.subplots_adjust(left=0.20, bottom=0.15, right=0.98, top=0.95)
plt.xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS - subtract)
plt.ylabel('time (ms)', fontsize=FONTSIZE_Y_AXIS - subtract - 4)
plt.yscale('log')
plt.grid(True)#, which="both", linestyle='--', linewidth=0.5)
plt.savefig(f"figures/nuc_ablation_ratio=2.pdf")

plt.figure(figsize=(8, 8))
plt.plot(all_m_r5, avg_vector_proj_times_nuc_r5, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.plot(all_m_r5, avg_matrix_proj_times_nuc_r5, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
plt.subplots_adjust(left=0.20, bottom=0.15, right=0.98, top=0.95)
plt.xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS - subtract)
plt.ylabel('time (ms)', fontsize=FONTSIZE_Y_AXIS - subtract - 4)
plt.yscale('log')
plt.grid(True)#, which="both", linestyle='--', linewidth=0.5)
plt.savefig(f"figures/nuc_ablation_ratio=5.pdf")

