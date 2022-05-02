from  matplotlib import pyplot as plt
import sys, os, numpy as np

plt.rcParams["font.size"] = 12 # 24

# PicassoMPM outputs
kernel_list = ['p2g_H', 'div_T', 'g2p_H', 'grid_T', 'update_H', 'BC_H', 'p2g_U', 'div_U', 'g2p_U', 'grid_U', 'update_U', 'BC_U', 'particle_S', 'surfaceTension', 'TreeTraversal::nearest']#, ' pack_buffer', 'unpack_buffer']
# CabanaMD outputs
#kernel_list = ['compute_full', 'fill_neighbors', 'compute_energy_full', 'update_halo', 'initial_integrate']
# CabanaPD outputs
#kernel_list = ['compute_full', 'compute_energy', 'VerletList', 'Integrator', 'gather']

# PD #color_list = ['#E31A1C', '#4291C7', '#248542', '#FF913E', '#9EC7E4', '#ADDE8F', '#BFBFBF', 'k']
color_list = ['darkgrey']*len(kernel_list)
color_dict = {'p2g': '#E31A1C', 'div': '#4291C7', 'g2p': '#248542', 'grid': '#FF913E', 'update': '#9EC7E4', 'BC': '#ADDE8F', '_S': "violet", "surface": "purple", "TreeTraversal": "pink"}
for k, kernel in enumerate(kernel_list):
    for key, val in color_dict.items():
        if key in kernel:
            color_list[k] = val
width = 0.05
filenames = sys.argv[1:]
n_files = len(filenames)

kernel_list.append('Time outside Kokkos')
kernel_list.append('Total Execution Time')
color_list.append('#BFBFBF')
color_list.append('k')
kernel_dict = {key: np.array([]) for key in kernel_list}
n_kernels = len(kernel_dict)
x_indices = np.arange(n_files)

names = ['mi250x', 'v100']
#names = ['V100', 'P9']
for fn, file in enumerate(filenames):
    #ranks = int(file.split('.')[0].split('_')[-1])
    #ranks = int(file.split('.')[0].split('_')[-3].split('n')[-1]) #PD
    ranks = 1
    print(ranks)
    #names.append(file.split('_')[0].split("/")[-1])# + " (" + str(ranks) + ")")
    with open(file) as f:
        txt = f.readlines()

    for l, line in enumerate(txt):
        for kernel in kernel_dict.keys():
            if kernel in line:# and not kernel_dict[kernel]:
                if 'outside' not in kernel and 'Total' not in kernel:
                    kernel_dict[kernel] = np.append(kernel_dict[kernel], float(txt[l+1].split()[1]) / ranks)
                else:
                    kernel_dict[kernel] = np.append(kernel_dict[kernel], float(txt[l].split()[-2]) / ranks)

    for kernel in kernel_dict.keys():
        print(kernel, kernel_dict[kernel][fn:])
        if not fn:
            kernel_dict[kernel] = np.array([np.sum(kernel_dict[kernel])])
        else:
            kernel_dict[kernel][fn] = np.sum(kernel_dict[kernel][fn:])
            kernel_dict[kernel] = np.resize(kernel_dict[kernel], fn+1)
    #print(kernel_dict)

# REMOVE
# CPU run only 1/10 total runtime
"""if len(filenames) > 1:
    for kernel in kernel_dict.keys():
        print(kernel, kernel_dict[kernel])
        if 'Verlet' not in kernel:
            kernel_dict[kernel][1] *= 10
        print(kernel, kernel_dict[kernel])
    kernel_dict['Total Execution Time'][1] -= kernel_dict["VerletList"][1]*9
"""
print(kernel_dict)

fig = plt.figure()
ax1 = fig.add_subplot(111)

step = 0.0
k = 0
labels = []
for key, val in kernel_dict.items():
    if len(val):
        label = key
        if 'outside' in key: label = 'non-Kokkos'
        elif 'Total' in key: label = 'Total'
        # Replace all MD labels
        if 'compute_full' in key: label = 'Force'
        if 'Verlet'in key: label = 'Neighbor'
        if 'compute_energy'in key: label = 'Energy'
        if 'gather' in key: label =  'Halo'
        if 'integrate'in key: label = 'Integrate'
        if 'TreeTraversal' in key: label = "ArborX"
        if 'surfaceTension' in key: label = "surface"
        plt.bar(x_indices+step, val / val[0],
                width,
                color=color_list[k],
                label=label)
        step += width
        labels.append(label)
    k += 1

#print(kernel_dict['Total Execution Time']/kernel_dict['Total Execution Time'][0])
#minor_x_indices = np.arange(-width/4, width*len(labels), width)
#plt.xticks(minor_x_indices, labels, rotation=45)
plt.xticks(x_indices+0.3, names)
ax1.set_ylabel('MI250X speedup')
#ax1.set_ylabel("Time (s)")

# Only if needed
box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.95, box.height])
#ax1.legend(loc='center right', bbox_to_anchor=(0.6, 0.55))
ax1.legend(bbox_to_anchor=(1, 1.0))

ax1.set_yscale('log')
ax1.set_xlim([0.9,1.9])
#ax1.set_ylim([5e-3,1e4])
fig.tight_layout()

plt.show()
#plt.savefig("plot.png", dpi=300)
