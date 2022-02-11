from  matplotlib import pyplot as plt
import sys, os, numpy as np

plt.rcParams["font.size"] = 12

# PicassoMPM outputs
kernel_list = ['p2g', 'divT', 'g2p', 'gridH', 'nextH', 'BC']
# CabanaMD outputs
kernel_list = ['compute_full', 'fill_neighbors', 'compute_energy_full', 'update_halo', 'initial_integrate']
color_list = ['#E31A1C', '#4291C7', '#248542', '#FF913E', '#9EC7E4', '#BFBFBF', 'k']
width = 0.1

filenames = sys.argv[1:]
n_files = len(filenames)

kernel_list.append('Time outside Kokkos')
kernel_list.append('Total Execution Time')
kernel_dict = {key: [] for key in kernel_list}
n_kernels = len(kernel_dict)
x_indices = np.arange(n_files)
names = []
for file in filenames:
    ranks = int(file.split('.')[0].split('_')[-1])
    names.append(file.split('_')[0])# + " (" + str(ranks) + ")")
    with open(file) as f:
        txt = f.readlines()

    for l, line in enumerate(txt):
        for kernel in kernel_dict.keys():
            if kernel in line:
                if 'outside' not in kernel and 'Total' not in kernel:
                    kernel_dict[kernel].append(float(txt[l+1].split()[1]) / ranks)
                else:
                    kernel_dict[kernel].append(float(txt[l].split()[-2]) / ranks)


fig = plt.figure()
ax1 = fig.add_subplot(111)

step = 0.0
k = 0
for key, val in kernel_dict.items():
    label = key
    if 'outside' in key: label = 'non-Kokkos'
    elif 'Total' in key: label = 'Total'
    # Replace all MD labels
    if 'compute_full' in key: label = 'Force'
    if 'fill_neighbors'in key: label = 'Neighbor'
    if 'compute_energy_full'in key: label = 'Energy'
    if 'update_halo'in key: label =  'Halo'
    if 'initial_integrate'in key: label = 'Integrate'
    plt.bar(x_indices+step, val, width, color=color_list[k], label=label)
    step += width
    k += 1

plt.xticks(x_indices+0.4, names)
ax1.set_ylabel("Time (s)")

# Only if needed
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))
#ax1.legend()
ax1.set_yscale('log')

plt.show()
#plt.savefig("plot.png", dpi=300)
