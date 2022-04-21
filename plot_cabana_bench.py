from  matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import sys, os, numpy as np

def skip_lines(txt, l):
    while l < len(txt) and not txt[l].isspace():
        l += 1
    return l

plt.rcParams["font.size"] = 12

all_backends = ['host', 'serial', 'openmp', 'cuda', 'hip', 'cudauvm']

backends = ['cuda', 'serial']
#backends = ['cuda_cuda', 'hip_hip']

size_list = [1e3, 1e4, 1e5, 1e6, 1e7]
size_list = [16, 32, 64, 128, 256]

#type_list = ['create', 'migrate']
#type_list = ['create', 'permute']
#type_list = ['create', 'iteration']
type_list = ['create', 'scatter']

param_list = ['halo']
#param_list = ['10', '100', '1000', '10000', '100000', '1000000', '10000000'] #['sort']
#param_list = ['3', '4', '5']

comm = 'dist' in param_list or 'halo' in param_list

color_dict = {type_list[0]: '#E31A1C', type_list[1]:'#4291C7'}
dash_dict = {backends[0]: "-", backends[1]: "--"}

width = 0.1
linewidth = 2

filenames = sys.argv[1:]
n_files = len(filenames)
n_sizes = len(size_list)

backend_dict = {key: {key: {key: [] for key in type_list} for key in param_list} for key in backends}
sizes_dict = {key: {key: {key: [] for key in type_list} for key in param_list} for key in backends}

for file in filenames:
    if comm:
        read_size = float(file.split("_")[-1])

    with open(file) as f:
        txt = f.readlines()

    l = 0
    while l < len(txt):
        if txt[l].isspace() or 'problem_size' in txt[l]:
            l += 1
        elif "Cabana Comm" in txt[l] or "Cajita Halo" in txt[l]:
            l = skip_lines(txt, l)

        elif txt[l].split("_")[0] in backends or "_".join(txt[l].split("_")[:2]) in backends:
            for backend in backends:
                if backend in txt[l]:
                    current_backend = backend
            current_param = None
            for param in param_list:
                if param in txt[l] and "." not in txt[l]:
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
            if not comm:
                read_size = float(vals[0])
            for s, size in enumerate(size_list):
                if size == read_size:
                    backend_dict[current_backend][current_param][current_type].append(float(vals[-1]))
                    sizes_dict[current_backend][current_param][current_type].append(size)
            l += 1

print(backend_dict)
#print(sizes_dict)

fig = plt.figure()
ax1 = fig.add_subplot(111)

# Plot all separate
'''for backend in backend_dict:
    for param in param_list:
        for type in type_list:
            plt.plot(size_list, backend_dict[backend][param][type] / 1e6, width, color=color_dict[type], label=type)
'''
for backend in backend_dict:
    for param in param_list:
        for type in type_list:
            #for size in size_list:
            y = np.array(backend_dict[backend][param][type]) #/ np.array(backend_dict[backends[1]][param][type])
            if "host" in backend or "serial" in backend or "openmp" in backend:
                plt.plot(np.array(sizes_dict[backend][param][type])*1.05, y, color=color_dict[type], lw=linewidth, linestyle="none", fillstyle="none", marker='o')# linestyle=dash_dict[backend])#, facecolors=color_dict[type])
            else:
                plt.plot(sizes_dict[backend][param][type], y, color=color_dict[type], lw=linewidth, linestyle="none", marker='o')

#ax1.set_ylabel("Speedup relative to V100-CUDA")
ax1.set_ylabel("Time (seconds)")
#ax1.set_xlabel("Number of particles")
ax1.set_xlabel("Number of grid points per dimension")
ax1.set_title("Cabana benchmark (ORNL Summit)")
fake_lines = [Line2D([0], [0], color=color_dict[type_list[0]], lw=2, label=type_list[0]),
              Line2D([0], [0], color=color_dict[type_list[1]], lw=2, label=type_list[1]),
              #Line2D([0], [0], color="k", lw=2, linestyle=dash_dict[backends[1]], label="POWER9 CPU"),
              #Line2D([0], [0], color="k", lw=2, linestyle=dash_dict[backends[0]], label="V100 GPU")
              Line2D([0], [0], color="k", lw=2, linestyle="none", fillstyle="none", marker='o', label="POWER9 CPU"),
              Line2D([0], [0], color="k", lw=2, linestyle="none", marker='o', label="V100 GPU")]

# Only if needed
ax1.legend(handles=fake_lines)
ax1.set_xscale('log')
ax1.set_yscale('log')

plt.show()
#plt.savefig("plot.png", dpi=300)
