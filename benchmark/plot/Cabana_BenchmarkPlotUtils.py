from  matplotlib import pyplot as plt
from matplotlib.lines import Line2D

import numpy as np
from copy import deepcopy

class DataDescription:
    def __init__(self, label):
        #Example: serial_neigh_iteration_3_1
        self.label = label
        details = label.split("_")
        self.backend = details[0].strip()
        self.type = details[1].strip()
        self.category = details[2].strip()
        # FIXME
        if self.category == "iteration": self.category = "iterate"
        self.params = []
        for p in details[3:]:
            self.params.append(p.strip())

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

class DataDescriptionInterpolation(DataDescription):
    # Purposely not calling base __init__
    def __init__(self, label):
        #Example: device_p2g_scalar_value_16
        self.label = label
        details = label.split("_")
        self.backend = details[0].strip()
        self.type = "interpolation"
        self.category = details[1].strip()
        self.params = ["_".join(details[2:4]).strip()]
        self.size = details[-1]

class ManualDataDescription:
    def __init__(self, backend, type, category, params):
        self.backend = backend
        self.type = type
        self.category = category
        self.params = params
        label_list = [backend, type, category] + params
        self.label = "_".join(label_list)

class DataPoint:
    num_rank = 1

    def __init__(self, description, line):
        self.description = description

        #problem_size min max ave
        self.line = line
        results = line.split()
        self.size = int(float(results[0]))
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

class DataPointInterpolation(DataPoint):
    # Purposely not calling base __init__
    def __init__(self, description, line):
        # Deep copy necessary because unique parameters are used per result (ppc)
        self.description = deepcopy(description)

        #ppc min max ave
        self.line = line
        results = line.split()
        self.size = int(float(description.size))
        self._initTimeResults(results[1:])
        self.description.params.append(results[0])
        print(self.size, self.ave, self.description.category, self.description.type, self.description.params)

class AllData:
    def __init__(self, filelist, grid=False):
        self.grid = grid
        self.results = []
        self.filelist = filelist
        for filename in filelist:
            self._readFile(filename)

    def _endOfFile(self, l):
        return l >= self.total

    def _emptyLine(self, line):
        if line.isspace():
            return True
        return False

    def _headerLine(self, line):
        if 'min max ave' in line:
            return True
        return False

    def _getDescription(self, line):
        return DataDescription(line)

    def _getData(self, descr, line):
        return DataPoint(descr, line)

    def _readFile(self, filename):
        with open(filename) as f:
            txt = f.readlines()
        l = 0
        self.total = len(txt)
        while not self._endOfFile(l):
            if self._emptyLine(txt[l]):
                l += 1
                description = self._getDescription(txt[l])
            elif self._headerLine(txt[l]):
                l += 1
                while not self._endOfFile(l) and not self._emptyLine(txt[l]):
                    self.results.append(self._getData(description, txt[l]))
                    l += 1
            else:
                l += 1

    def minMaxSize(self):
        min = 1e100
        max = -1
        for r in self.results:
            if r.size < min: min = r.size
            if r.size > max: max = r.size
        return np.array([min, max])

    def getAllBackends(self):
        unique = []
        for r in self.results:
            backend = r.description.backend
            if backend not in unique:
                unique.append(backend)
        return unique

    def getAllTypes(self):
        unique = []
        for r in self.results:
            type = r.description.type
            if type not in unique:
                unique.append(type)
        return unique

    def getAllParams(self):
        unique = []
        for r in self.results:
            params = r.description.params
            if params not in unique:
                unique.append(params)
        return unique

    def getAllCategories(self):
        unique = []
        for r in self.results:
            category = r.description.category
            if category not in unique:
                unique.append(category)
        return unique

class AllDataMPI(AllData):
    def _endOfFile(self, l):
        return l >= self.total

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

    def getAllCommFractions(self):
        unique = []
        for r in self.results:
            send = r.send_bytes_n
            if send not in unique:
                unique.append(send)
        return unique

class AllDataInterpolation(AllData):
    def _getDescription(self, line):
        return DataDescriptionInterpolation(line)

    def _getData(self, descr, line):
        return DataPointInterpolation(descr, line)

class AllSizesSingleResult:
    def __init__(self, all_data: AllData, descr: ManualDataDescription):
        self.data = []
        self.sizes = []
        for d in all_data.results:
            if self._compareAll(d.description, descr):
                self.sizes.append(d.size)
                self.data.append(d.ave)
        print(self.sizes, self.data)
    def _compareAll(self, data_description, check):
        if data_description.backend == check.backend and data_description.category == check.category and data_description.type == check.type and data_description.params == check.params:
            return True
        return False

class AllSizesSingleResultMPI:
    def __init__(self, all_data: AllDataMPI, descr: ManualDataDescription, n):
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


def getColors(data: AllData):
    color_list = ["#e31a1c", "#1f78b4", "#a6cee3", "#cab2d6", "#fdbf6f", "#ffff99"]
    color_dict = {}
    categories = data.getAllCategories()
    for c, cat in enumerate(categories):
        color_dict[cat] = color_list[c]

    return color_dict

def getLegend(data: AllData, cpu_name, gpu_name, speedup):

    legend = []
    # No way to know what's being compared here.
    if not speedup:
        backends = data.getAllBackends()
        for backend in data.getAllBackends():
            # FIXME: backwards compatibility
            if "host" in backend or "serial" in backend or "openmp" in backend:
                legend.append(Line2D([0], [0], color="k", lw=2, linestyle= "--", label=cpu_name+" CPU"))
            if "device" in backend or "cuda" in backend or "hip" in backend:
                legend.append(Line2D([0], [0], color="k", lw=2, linestyle="-", label=gpu_name+" GPU"))

    colors = getColors(data)
    categories = data.getAllCategories()
    for cat in categories:
        legend.append(Line2D([0], [0], color=colors[cat], lw=2, label=cat))
    return legend

def plotResults(fig, ax, x, y, backend, color):
    linewidth = 2
    dash = "-"
    offset = 1.0
    # FIXME: backwards compatibility
    if "host" in backend or "serial" in backend or "openmp" in backend:
        dash = "--"
        offset = 1.1

    ax.plot(x*offset, y, color=color, lw=linewidth, marker='o', linestyle=dash)

def createPlot(fig, ax, data: AllData, speedup=False, cpu_name="", gpu_name=""):
    if speedup:
        min_max = data.minMaxSize()
        if data.grid: min_max = min_max**3
        ax.plot(min_max, [1]*len(min_max), c="k")

    fig.tight_layout()
    plt.rcParams["font.size"] = 12

    if speedup:
        ax.set_ylabel("Speedup")
    else:
        ax.set_ylabel("Time (seconds)")
    if data.grid:
        ax.set_xlabel("Number of grid points")
    else:
        ax.set_xlabel("Number of particles")

    lines = getLegend(data, cpu_name, gpu_name, speedup)
    ax.legend(handles=lines)
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.show()
    #plt.savefig("Cabana_Benchmark.png", dpi=300)
