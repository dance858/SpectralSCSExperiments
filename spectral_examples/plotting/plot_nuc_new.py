import numpy as np
import seaborn as sns 
import pdb
import matplotlib.pyplot as plt
from SIZES_FIGURES import MARKER_SIZE, FONTSIZE_Y_AXIS, LINEWIDTH, FONTSIZE_X_AXIS, LEGEND_SIZE, TICK_SIZE_X, TICK_SIZE_Y
from matplotlib.ticker import ScalarFormatter, LogFormatter

sns.set_theme(style="whitegrid")

SUBTRACT = 14
SUBTRACT2 = 5

def returnAverageStatsLogDet(total_times_logdet, iters_logdet,
                             spectral_vector_proj_times, spectral_matrix_proj_times,
                             PSD_total_times, iters_PSD, PSD_cone_matrix_proj_times):
    average_solve_time = np.mean(total_times_logdet, axis=0)
    average_iters_logdet = np.mean(iters_logdet, axis=0)
    average_spectral_vector_proj_times = np.mean(spectral_vector_proj_times, axis=0)
    average_spectral_matrix_proj_times = np.mean(spectral_matrix_proj_times, axis=0)
    average_PSD_solve_time = np.mean(PSD_total_times, axis=0)
    average_PSD_iter = np.mean(iters_PSD, axis=0)
    average_PSD_cone_matrix_proj_time = np.mean(PSD_cone_matrix_proj_times, axis=0)

    return average_solve_time, average_iters_logdet, average_spectral_vector_proj_times, \
           average_spectral_matrix_proj_times,average_PSD_solve_time, average_PSD_iter,  \
           average_PSD_cone_matrix_proj_time


# For ratio = 1
ratio = 1
total_times_logdet = np.array([[8.51, 8.97, 20.1, 48.9, 80.5, 126],
                               [7.35, 9.66, 25.7, 49.4, 75.9, 117],
                               [11.0, 8.72, 26.1, 49.0, 85.8, 132],
                               [5.34, 70.1, 21.5, 48.6, 80.0, 126], 
                               [4.56, 8.71, 21.2, 47.3, 86.4, 127]])
iters_logdet = np.array([[900,  325,  325, 425, 425, 425],
                         [875,  350,  425, 425, 400, 400],
                         [1350, 325,  425, 425, 450, 450],
                         [650,  2475, 350, 425, 425, 450], 
                         [550,  325,  350, 425, 450, 450]])
spectral_vector_proj_times = np.array([[3.82*1e-4, 3.24*1e-4, 3.66*1e-4, 3.97*1e-4, 4.77*1e-4, 5.17*1e-4],
                                       [2.98*1e-4, 3.47*1e-4, 3.58*1e-4, 4.69*1e-4, 4.69*1e-4, 5.19*1e-4],
                                       [3.09*1e-4, 3.41*1e-4, 3.61*1e-4, 4.18*1e-4, 4.52*1e-4, 4.9*1e-4],
                                       [3.13*1e-4, 3.08*1e-4, 3.52*1e-4, 4.05*1e-4, 4.59*1e-4, 5.1*1e-4],
                                       [3.59*1e-4, 3.59*1e-4, 3.64*1e-4, 4.16*1e-4, 4.49*1e-4, 5.1*1e-4]])
spectral_matrix_proj_times = np.array([[7.51, 22.5, 52.8, 100.5, 168, 268],
                                       [6.52, 22.6, 51.9, 101.4, 169, 264],
                                       [6.29, 21.8, 52.4, 103,   169, 264],
                                       [6.30, 23.8, 52.8, 99.8,  167, 267],
                                       [6.33, 21.7, 51.6, 96.7,  171, 268]])
PSD_total_times = np.array([[15.7, 133,  598, 1330, 3040, 5250],
                            [11.1, 189,  719, 1080, 2860, 4480],
                            [20.0, 336,  180, 1350, 2900, 5780],
                            [14.3,  68,  551, 1490, 2490, 4600],
                            [13.9, 155, 1540, 1470, 1980, 4700]])
iters_PSD = np.array([[675, 1900,  3900, 4575, 6325, 7075],
                      [475, 2750,  4700, 3775, 5975, 6025],
                      [875, 4850,  1150, 4725, 6050, 7750],
                      [625, 975,   3575, 5200, 5175, 6325 ],
                      [600, 2225, 10000, 5175, 4125, 6325]])
PSD_cone_matrix_proj_times = np.array([[20.8, 64.1, 142, 271, 453, 704],
                                       [20.9, 63.1, 141, 269, 452, 706],
                                       [20.4, 63.5, 143, 268, 453, 709],
                                       [20.5, 64.8, 143, 268, 454, 707],
                                       [20.8, 63.6, 143, 266, 453, 708]])


average_solve_time_r1, average_iters_logdet_r1, average_spectral_vector_proj_times_r1, \
average_spectral_matrix_proj_times_r1, average_PSD_solve_time_r1, average_PSD_iter_r1,  \
average_PSD_cone_matrix_proj_time_r1 = returnAverageStatsLogDet(total_times_logdet, iters_logdet,
                             spectral_vector_proj_times, spectral_matrix_proj_times,
                             PSD_total_times, iters_PSD, PSD_cone_matrix_proj_times)

all_m_r1 = np.array([50, 100, 150, 200, 250, 300])
all_n_r1 = all_m_r1 / ratio 

# For ratio = 2.
ratio = 2
total_times_logdet = np.array([[1.51, 4.28, 5.23, 12, 20.6,   33.8, 54.4], 
                               [1.04, 14.9, 6.67, 12.1, 20.5, 32.1, 49.9],
                               [0.85, 1.85, 5.14, 10.6, 17.0, 24.7, 46.3],
                               [0.97, 12.5, 5.15, 10.1, 20.5, 26.4, 51.3]])
iters_logdet = np.array([[575, 625, 325, 425, 425, 450, 500],
                         [425, 2100, 425, 425, 425, 425, 450],
                         [350, 250, 325, 375, 350, 325, 425],
                         [400, 1775, 325, 350, 425, 325, 450]])

spectral_vector_proj_times = np.array([[2.74*1e-4, 3.24*1e-4, 3.24*1e-4, 3.28*1e-4, 3.80*1e-4, 4.16*1e-4, 4.34*1e-4],
                                       [2.52*1e-4, 3.10*1e-4, 3.28*1e-4, 4.04*1e-4, 4.17*1e-4, 4.31*1e-4, 4.6*1e-4],
                                       [2.38*1e-4, 4.86*1e-4, 3.33*1e-4, 3.49*1e-4, 3.88*1e-4, 4.31*1e-4, 4.39*1e-4 ],
                                       [2.55*1e-4, 3.25*1e-4, 3.34*1e-4, 3.81*1e-4, 3.91*1e-4, 5.94*1e-4, 4.38*1e-4 ]])

spectral_matrix_proj_times = np.array([[1.68, 4.76, 11.6, 21.6, 38.4, 61.1, 89],
                                       [1.61, 5.05, 11.5, 21.5, 37.8, 60.7, 91.9],
                                       [1.55, 4.83, 11.4, 21.5, 38.1, 61.3, 89.6],
                                       [1.58, 5.02, 11.4, 21.7, 38.2, 64.6, 91.7]])



PSD_total_times = np.array([[5.72, 50.5, 163, 376, 551, 956, 2730],
                            [4.10, 70.8, 298, 253, 563, 1480, 2180],
                            [3.1, 10.8, 150, 856, 539, 1350, 1880],
                            [3.95, 26.2, 59, 814, 2040, 1360, 1530]])
iters_PSD = np.array([[625, 1925, 2750, 3425, 2975, 3325, 6225],
                      [450, 2675, 5075, 2300, 3050, 5175, 5200],
                      [325, 400, 2550, 7875, 2925, 4525, 4500],
                      [425, 975, 1000, 7400, 10925, 4725, 3650]])

PSD_cone_matrix_proj_times = np.array([[7.99, 23.4, 53.9, 101,   172, 269, 413],
                                       [7.93, 23.6, 53.3, 101,   171, 268, 394],
                                       [8.1,  24.1, 53.6, 99.7,  171, 281, 393],
                                       [8.1,  24.1, 54.6, 101.4, 173, 268, 394]])

average_solve_time_r2, average_iters_logdet_r2, average_spectral_vector_proj_times_r2, \
average_spectral_matrix_proj_times_r2, average_PSD_solve_time_r2, average_PSD_iter_r2,  \
average_PSD_cone_matrix_proj_time_r2 = returnAverageStatsLogDet(total_times_logdet, iters_logdet,
                             spectral_vector_proj_times, spectral_matrix_proj_times,
                             PSD_total_times, iters_PSD, PSD_cone_matrix_proj_times)
all_m_r2 = np.array([50, 100, 150, 200, 250, 300, 350])
all_n_r2 = all_m_r2/ratio

# For ratio = 5. total_times in seconds. Projection times in milliseconds
ratio = 5
total_times_logdet = np.array([[0.159, 0.403, 0.729, 1.84, 2.36, 4.47, 5.82],
                               [0.134, 0.729, 0.538, 2.15, 1.41, 4.37, 5.76], 
                               [0.105, 0.404,  1.74,  9.3, 2.35, 4.07, 5.75], 
                               [0.12,  0.576,  1.01, 1.79, 1.96, 5.24, 5.75]])
iters_logdet = np.array([[275, 275, 250, 350,  275, 350, 325],
                         [225, 300, 175, 400,  150, 350, 325], 
                         [175, 275, 600, 1825, 275, 325, 325], 
                         [200, 375, 350, 325,  225, 425, 325]])
spectral_vector_proj_times =  np.array([[3.33*1e-4, 4.46*1e-4, 2.31*1e-4, 2.73*1e-4, 3.27*1e-4, 3.36*1e-4, 3.6*1e-4],
                                        [3.61*1e-4, 4.59*1e-4, 2.33*1e-4, 3.02*1e-4, 3.45*1e-4, 3.24*1e-4, 3.36*1e-4], 
                                        [3.54*1e-4, 2.2*1e-4,  2.26*1e-4, 2.79*1e-4, 3.43*1e-4, 3.31*1e-4, 3.75*1e-4], 
                                        [3.44*1e-4, 2.3*1e-4,  2.4*1e-4,  3.38*1e-4, 3.26*1e-4, 3.09*1e-4, 3.43*1e-4]])
spectral_matrix_proj_times = np.array([[2.23*1e-1, 6.5*1e-1,  1.42, 2.79, 4.57, 7.11, 10.4],
                                       [2.29*1e-1, 9.95*1e-1, 1.41, 2.81, 4.56, 7.08, 10.5], 
                                       [2.28*1e-1, 6.55*1e-1, 1.46, 2.77, 4.53, 7.06, 10.5], 
                                       [2.26*1e-1, 6.85*1e-1, 1.41, 2.81, 4.56, 7.08, 10.4]])
PSD_total_times = np.array([[1.42, 3.91, 7.11, 18.2, 76.5, 252, 444],
                            [1.05, 5.51, 7.83, 38.2, 87.4, 177, 289],
                            [0.94, 4.75, 18.7, 37.5, 107, 178, 209],
                            [1.05, 4.87, 6.69, 65.4, 63.6, 210, 562]])
iters_PSD = np.array([[300, 275, 225, 325, 850, 1850, 2350],
                      [225, 325, 250, 700, 975, 1300, 1450], 
                      [200, 325, 625, 675, 1225, 1300, 1050 ], 
                      [225, 350, 200, 1225, 700, 1550, 2900]])
PSD_cone_matrix_proj_times = np.array([[4.14, 12.7, 28.7, 51.5, 83.6, 127, 177],
                                       [4.02, 14.9, 28.5, 50.3, 83.2, 127, 187],
                                       [4.07, 13.1, 27.4, 51.9, 81.3, 128, 187],
                                       [4.05, 12.5, 30.3, 49.2, 84.3, 126, 182]])
all_m_r5 = np.array([50, 100, 150, 200, 250, 300, 350])
all_n_r5 = all_m_r5 / ratio 


average_solve_time_r5, average_iters_logdet_r5, average_spectral_vector_proj_times_r5, \
average_spectral_matrix_proj_times_r5, average_PSD_solve_time_r5, average_PSD_iter_r5,  \
average_PSD_cone_matrix_proj_time_r5 = returnAverageStatsLogDet(total_times_logdet, iters_logdet,
                             spectral_vector_proj_times, spectral_matrix_proj_times,
                             PSD_total_times, iters_PSD, PSD_cone_matrix_proj_times)





speedup1 = average_PSD_solve_time_r1 / average_solve_time_r1
speedup2 = average_PSD_solve_time_r2 / average_solve_time_r2
speedup5 = average_PSD_solve_time_r5 / average_solve_time_r5
print("speedup1: ", speedup1)
print("speedup2: ", speedup2)
print("speedup5: ", speedup5)
print("average speedup: ", np.mean([np.mean(speedup1), np.mean(speedup2), np.mean(speedup5)]))


#
average_logdet_iter_time_r1 = average_solve_time_r1 / average_iters_logdet_r1
average_logdet_iter_time_r2 = average_solve_time_r2 / average_iters_logdet_r2
average_logdet_iter_time_r5 = average_solve_time_r5 / average_iters_logdet_r5 

average_standard_iter_time_r1 = average_PSD_solve_time_r1 / average_PSD_iter_r1
average_standard_iter_time_r2 = average_PSD_solve_time_r2 / average_PSD_iter_r2
average_standard_iter_time_r5 = average_PSD_solve_time_r5 / average_PSD_iter_r5 

speedup1_iter = average_standard_iter_time_r1 / average_logdet_iter_time_r1
speedup2_iter = average_standard_iter_time_r2 / average_logdet_iter_time_r2
speedup5_iter = average_standard_iter_time_r5 / average_logdet_iter_time_r5

print("speedup1_iter average: ", np.mean(speedup1_iter))
print("speedup2_iter average: ", np.mean(speedup2_iter))
print("speedup5_iter average: ", np.mean(speedup5_iter))


FONTSIZE_X_AXIS += 5 
FONTSIZE_Y_AXIS += 8

# ------------------------------------------------------------------------------
#                               For ratio = 1
# ------------------------------------------------------------------------------
fig, (ax1, ax2, ax3) = plt.subplots(3, 4, figsize=(32, 25), sharey=False, sharex=True)
ax1[0].plot(all_m_r1, average_solve_time_r1, label='SpectralSCS', marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[0].plot(all_m_r1, average_PSD_solve_time_r1, label='SCS', marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[0].set_ylabel("runtime (s)", fontsize=FONTSIZE_Y_AXIS - 13)
ax1[0].set_yscale('log')
ax1[0].grid(True)

ax1[1].plot(all_m_r1, average_iters_logdet_r1, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[1].plot(all_m_r1, average_PSD_iter_r1, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[1].grid(True)
ax1[1].set_ylabel("iterations", fontsize=FONTSIZE_Y_AXIS - 13)

ax1[2].plot(all_m_r1, 1000*average_solve_time_r1 / average_iters_logdet_r1, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[2].plot(all_m_r1, 1000*average_PSD_solve_time_r1 / average_PSD_iter_r1, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[2].grid(True)
ax1[2].set_ylabel("iteration time (ms)", fontsize=FONTSIZE_Y_AXIS - 13)

ax1[3].plot(all_m_r1, average_spectral_matrix_proj_times_r1, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax1[3].plot(all_m_r1, average_PSD_cone_matrix_proj_time_r1, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
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
ax2[0].plot(all_m_r2, average_solve_time_r2, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[0].plot(all_m_r2, average_PSD_solve_time_r2, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[0].set_ylabel("runtime (s)", fontsize=FONTSIZE_Y_AXIS - 13)
ax2[0].set_yscale('log')
ax2[0].grid(True)

ax2[1].plot(all_m_r2, average_iters_logdet_r2, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[1].plot(all_m_r2, average_PSD_iter_r2, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[1].grid(True)
ax2[1].set_ylabel("iterations", fontsize=FONTSIZE_Y_AXIS - 13)

ax2[2].plot(all_m_r2, 1000*average_solve_time_r2 / average_iters_logdet_r2, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[2].plot(all_m_r2, 1000*average_PSD_solve_time_r2 / average_PSD_iter_r2, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[2].grid(True)
ax2[2].set_ylabel("iteration time (ms)", fontsize=FONTSIZE_Y_AXIS - 13)

ax2[3].plot(all_m_r2, average_spectral_matrix_proj_times_r2, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[3].plot(all_m_r2, average_PSD_cone_matrix_proj_time_r2, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax2[3].grid(True)
ax2[3].set_ylabel("matrix cone projection time (ms)", fontsize=FONTSIZE_X_AXIS - 24)

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
ax3[0].plot(all_m_r5, average_solve_time_r5, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[0].plot(all_m_r5, average_PSD_solve_time_r5, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[0].set_ylabel("runtime (s)", fontsize=FONTSIZE_Y_AXIS - 13)
ax3[0].set_yscale('log')
ax3[0].grid(True)

ax3[1].plot(all_m_r5, average_iters_logdet_r5, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[1].plot(all_m_r5, average_PSD_iter_r5, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[1].grid(True)
ax3[1].set_ylabel("iterations", fontsize=FONTSIZE_Y_AXIS - 13)

ax3[2].plot(all_m_r5, 1000*average_solve_time_r5 / average_iters_logdet_r5, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[2].plot(all_m_r5, 1000*average_PSD_solve_time_r5 / average_PSD_iter_r5, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[2].grid(True)
ax3[2].set_ylabel("iteration time (ms)", fontsize=FONTSIZE_Y_AXIS - 13)

ax3[3].plot(all_m_r5, average_spectral_matrix_proj_times_r5, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[3].plot(all_m_r5, average_PSD_cone_matrix_proj_time_r5, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
ax3[3].grid(True)
ax3[3].set_ylabel("matrix cone projection time (ms)", fontsize=FONTSIZE_X_AXIS - 24)

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

plt.savefig("robust_pca.pdf") 

pdb.set_trace()

# -------------------------------------
# Ablation
# ------------------------------------
subtract = 3
plt.figure(figsize=(8, 8))
plt.plot(all_m_r1, average_spectral_vector_proj_times_r1, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.plot(all_m_r1, average_spectral_matrix_proj_times_r1, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
plt.subplots_adjust(left=0.20, bottom=0.15, right=0.98, top=0.95)
plt.xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS - subtract)
plt.ylabel('time (ms)', fontsize=FONTSIZE_Y_AXIS - subtract - 4)
plt.yscale('log')
plt.grid(True)#, which="both", linestyle='--', linewidth=0.5)
plt.savefig(f"figures/nuc_ablation_ratio=1.pdf")

plt.figure(figsize=(8, 8))
plt.plot(all_m_r2, average_spectral_vector_proj_times_r2, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.plot(all_m_r2, average_spectral_matrix_proj_times_r2, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
plt.subplots_adjust(left=0.20, bottom=0.15, right=0.98, top=0.95)
plt.xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS - subtract)
plt.ylabel('time (ms)', fontsize=FONTSIZE_Y_AXIS - subtract - 4)
plt.yscale('log')
plt.grid(True)#, which="both", linestyle='--', linewidth=0.5)
plt.savefig(f"figures/nuc_ablation_ratio=2.pdf")

plt.figure(figsize=(8, 8))
plt.plot(all_m_r5, average_spectral_vector_proj_times_r5, marker='o', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.plot(all_m_r5, average_spectral_matrix_proj_times_r5, marker='s', linestyle = "--", markersize=MARKER_SIZE, linewidth=LINEWIDTH)
plt.tick_params(axis='both', which='major', labelsize=TICK_SIZE_Y)
plt.subplots_adjust(left=0.20, bottom=0.15, right=0.98, top=0.95)
plt.xlabel(r'$n$', fontsize=FONTSIZE_X_AXIS - subtract)
plt.ylabel('time (ms)', fontsize=FONTSIZE_Y_AXIS - subtract - 4)
plt.yscale('log')
plt.grid(True)#, which="both", linestyle='--', linewidth=0.5)
plt.savefig(f"figures/nuc_ablation_ratio=5.pdf")

average_spectral_matrix_proj_times_r5