from  matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import sys, os, numpy as np

def skip_lines(txt, l):
    while l < len(txt) and not txt[l].isspace():
        l += 1
    return l

plt.rcParams["font.size"] = 12

all_backends = ['host', 'serial', 'openmp', 'cuda', 'hip', 'cudauvm']
backends = ['cuda', 'hip']
#backends = ['cuda_cuda', 'hip_hip']

size_list = [1e3, 1e4, 1e5, 1e6, 1e7]
#type_list = ['create', 'gather']
#type_list = ['create', 'permute']
type_list = ['create', 'iteration']

#param_list = ['halo']
#param_list = ['10', '100', '1000', '10000']
param_list = ['3', '4']


color_dict = {type_list[0]: '#E31A1C', type_list[1]:'#4291C7'}
width = 0.1

filenames = sys.argv[1:]
n_files = len(filenames)
n_sizes = len(size_list)

backend_dict = {key: {key: {key: np.zeros([n_sizes]) for key in type_list} for key in param_list} for key in backends}

for file in filenames:
    with open(file) as f:
        txt = f.readlines()

    l = 0
    while l < len(txt):
        print(l)
        if txt[l].isspace() or 'problem_size' in txt[l]:
            l += 1
        elif "Cabana Comm" in txt[l]:
            l = skip_lines(txt, l)

        elif txt[l].split("_")[0] in backends or "_".join(txt[l].split("_")[:2]) in backends:
            for backend in backends:
                if backend in txt[l]:
                    current_backend = backend
            print(current_backend)
            current_param = None
            for param in param_list:
                if param in txt[l]:
                    current_param = param
            for type in type_list:
                if type in txt[l]:
                    current_type = type

            if current_param == None:
                l = skip_lines(txt, l)
            else:
                l += 2

        elif txt[l].split('_')[0] in all_backends:
            l = skip_lines(txt, l)

        else:
            vals = txt[l].split()
            read_size = float(vals[0])
            for s, size in enumerate(size_list):
                if size == read_size:
                    backend_dict[current_backend][current_param][current_type][s] = float(vals[-1])
            l += 1

print(backend_dict)

fig = plt.figure()
ax1 = fig.add_subplot(111)

# Plot all separate
'''for backend in backend_dict:
    for param in param_list:
        for type in type_list:
            plt.plot(size_list, backend_dict[backend][param][type] / 1e6, width, color=color_dict[type], label=type)
'''
for param in param_list:
    for type in type_list:
        y = backend_dict[backends[0]][param][type] / backend_dict[backends[1]][param][type]
        plt.plot(size_list, y, width, color=color_dict[type], lw=2)

ax1.set_ylabel("Time (s)")

fake_lines = [Line2D([0], [0], color=color_dict[type_list[0]], lw=2, label=type_list[0]),
              Line2D([0], [0], color=color_dict[type_list[1]], lw=2, label=type_list[1])]

# Only if needed
ax1.legend(handles=fake_lines)
ax1.set_xscale('log')

plt.show()
#plt.savefig("plot.png", dpi=300)
