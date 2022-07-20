from  matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import sys, numpy as np

class DataDescription:
    def __init__(self, label):
        #Example: serial_neigh_iteration_3_1
        self.label = label
        details = label.split("_")
        self.backend = details[0].strip()
        self.type = details[1].strip()
        self.category = details[2].strip()
        self.params = details[3:]

    def __init__(self, backend, type, category, params):
        self.backend = backend
        self.type = type
        self.category = category
        self.params = params
        label_list = [backend, type, category] + params
        self.label = "_".join(label_list)

class DataDescriptionMPI(DataDescription):
    # Purposely not calling base __init__
    def __init__(self, label):
        #Example: device_device_distributor_fast_create
        self.label = label
        details = label.split("_")
        self.backend = "_".join(details[0:2]).strip()
        self.type = details[2].strip()
        self.category = details[-1].strip()
        self.params = details[3:-1]

class DataPoint:
    num_rank = 1

    def __init__(self, description, line):
        self.description = description

        #problem_size min max ave
        self.line = line
        results = line.split()
        self.size = int(results[0])
        self._initTimeResults(results[1:])

    def _initTimeResults(self, results):
        self.min = float(results[0])
        self.max = float(results[1])
        self.ave = float(results[2])

class DataPointMPI(DataPoint):
    # Purposely not calling base __init__
    def __init__(self, description, line, n, size):
        self.description = description

        #num_rank send_bytes min max ave
        self.size = size
        self.line = line
        results = line.split()
        self.num_rank = int(results[0])
        self.send_bytes = float(results[1])
        self.send_bytes_n = n
        self._initTimeResults(results[2:])

class AllData:
    def __init__(self, filelist):
        self.results = []
        self.filelist = filelist
        for filename in filelist:
            self._readFile(filename)

    def _endOfFile(self, l):
        return l > self.total

    def _emptyLine(self, line):
        if line.isspace():
            return True
        return False

    def _headerLine(self, line):
        if 'problem_size' in line:
            return True
        return False

    def _readFile(self, filename):
        with open(filename) as f:
            txt = f.readlines()

        l = 0
        self.total = len(txt)
        while not self._endOfFile(l):
            if self._emptyLine(txt[l]):
                l += 1
                description = DataDescription(txt[l])
            elif self._headerLine(txt[l]):
                l += 1
                while not self._emptyLine(txt[l]) and not self._endOfFile(l):
                    self.results.append(DataPoint(description, txt[l]))
                    l += 1

    def minMaxSize(self):
        min = 1e100
        max = -1
        for r in self.results:
            if r.size < min: min = r.size
            if r.size > max: max = r.size
        return [min, max]

class AllDataMPI(AllData):
    def _endOfFile(self, l):
        return l > self.total

    def _headerLine(self, line):
        if 'num_rank' in line:
            return True
        return False

    def _readFile(self, filename):
        with open(filename) as f:
            txt = f.readlines()
        size = int(txt[4].split()[-1])

        l = 8
        self.total = len(txt[l:])
        while not self._endOfFile(l):
            if self._emptyLine(txt[l]):
                l += 1
                description = DataDescriptionMPI(txt[l])
            elif self._headerLine(txt[l]):
                l += 1
                n = 0
                while not self._emptyLine(txt[l]) and not self._endOfFile(l):
                    self.results.append(DataPointMPI(description, txt[l], n, size))
                    l += 1
                    n += 1
            else:
                l += 1

class AllSizesSingleResult:
    def __init__(self, all_data: AllData, descr: DataDescription, n):
        self.data = []
        self.sizes = []
        for d in all_data.results:
            if self._compareAll(d.description, descr, n) and n == d.send_bytes_n:
                self.sizes.append(d.size)
                self.data.append(d.ave)

    def _compareAll(self, data_description, check, n):
        if data_description.backend == check.backend and data_description.category == check.category and data_description.type == check.type and data_description.params == check.params:
            return True
        return False

plt.rcParams["font.size"] = 12

width = 0.1
linewidth = 2

#outname = sys.argv[1]
filelist = sys.argv[1:]
print(filelist)
data = AllDataMPI(filelist)

fig = plt.figure()
ax1 = fig.add_subplot(111)

backends = ['device_device', 'host_host']
type_list = ['create','gather']
color_dict = {type_list[0]: '#E31A1C', type_list[1]:'#4291C7'}#, type_list[2]: 'k'}
dash_dict = {backends[0]: "-", backends[1]: "--"}
offset_dict = {backends[0]: 1.0, backends[1]: 1.1}

for backend in ["device_device"]:#, "host_host"]:
    for type in ["gather", "scatter"]:
        print(backend, type)
        for n in range(7):
            prev = DataDescription(backend, "halo", type, ["slice"])
            prev_results = AllSizesSingleResult(data, prev, n)

            new = DataDescription(backend, "halo", type, ["buffer","slice"])
            new_results = AllSizesSingleResult(data, new, n)

            x = np.array(prev_results.sizes)*offset_dict[backend]
            print(prev_results.data, new_results.data)
            y = np.array(prev_results.data) / np.array(new_results.data)
            plt.plot(x, y, color=color_dict['gather'], lw=linewidth, marker='o',
                     linestyle=dash_dict[backend]) #, markerfacecolor=face_dict[type])

x = data.minMaxSize()
plt.plot(x, [1]*len(x), c="k")

#ax1.set_ylabel("MI250X-HIP speedup relative to V100-CUDA")
ax1.set_ylabel("Speedup")
#ax1.set_ylabel("Time (seconds)")
ax1.set_xlabel("Number of particles")
#ax1.set_xlabel("Number of grid points")
#ax1.set_title("Cabana benchmark - OLCF comparison")
fake_lines = [Line2D([0], [0], color=color_dict[type_list[0]], lw=2, label=type_list[0]),
              Line2D([0], [0], color=color_dict[type_list[1]], lw=2, label="iterate" if "iteration" in type_list[1] else type_list[1]),
              Line2D([0], [0], color="k", lw=2, linestyle=dash_dict[backends[1]], label="POWER9 CPU"),
              Line2D([0], [0], color="k", lw=2, linestyle=dash_dict[backends[0]], label="V100 GPU")]

# Only if needed
ax1.legend(handles=fake_lines)
ax1.set_xscale('log')
#ax1.set_yscale('log')
#ax1.set_ylim([0.1, 5])
fig.tight_layout()

plt.show()
#plt.savefig(outname+".png", dpi=300)



type_list = ['gather'] #['create', 'scatter']
#type_list = ['create', 'migrate']
#type_list = ['p2g', 'g2p']
#type_list = ['create', 'permute']
#type_list = ['create', 'iteration']#, 'build']

param_list = ['halo'] # dist
#param_list = ['16', '32', '64', '128', '256']
#param_list = ['', 'aosoa', 'slice']
#param_list =  ['10', '100', '1000', '10000', '100000', '1000000', '10000000'] #['sort']
#param_list = ['3', '4', '5']
