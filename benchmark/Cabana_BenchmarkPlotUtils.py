from  matplotlib import pyplot as plt
from matplotlib.lines import Line2D

class DataDescription:
    def __init__(self, label):
        #Example: serial_neigh_iteration_3_1
        self.label = label
        details = label.split("_")
        self.backend = details[0].strip()
        self.type = details[1].strip()
        self.category = details[2].strip()
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

class AllData:
    def __init__(self, filelist):
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
                while not self._endOfFile(l) and not self._emptyLine(txt[l]):
                    self.results.append(DataPoint(description, txt[l]))
                    l += 1
            else:
                l += 1

    def minMaxSize(self):
        min = 1e100
        max = -1
        for r in self.results:
            if r.size < min: min = r.size
            if r.size > max: max = r.size
        return [min, max]

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

    def getAllCommFractions(self):
        unique = []
        for r in self.results:
            send = r.send_bytes_n
            if send not in unique:
                unique.append(send)
        return unique

class AllSizesSingleResult:
    def __init__(self, all_data: AllData, descr: ManualDataDescription):
        self.data = []
        self.sizes = []
        for d in all_data.results:
            if self._compareAll(d.description, descr):
                self.sizes.append(d.size)
                self.data.append(d.ave)

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

def plotResults(fig, ax, x, y, backend, color):
    linewidth = 2
    dash = "-"
    offset = 1.0
    if "host" in backend or "serial" in backend or "openmp" in backend:
        dash = "--"
        offset = 1.1

    ax.plot(x*offset, y, color=color, lw=linewidth, marker='o', linestyle=dash)

def createPlot(fig, ax, min_max, speedup=False, grid=False):
    ax.plot(min_max, [1]*len(min_max), c="k")

    fake_lines = [Line2D([0], [0], color='#E31A1C', lw=2, label='create'),
                  Line2D([0], [0], color='#4291C7', lw=2, label="iterate"), # if "iteration" in type_list[1] else type_list[1]),
                  Line2D([0], [0], color="k", lw=2, linestyle= "--", label="POWER9 CPU"),
                  Line2D([0], [0], color="k", lw=2, linestyle="-", label="V100 GPU")]

    fig.tight_layout()
    plt.rcParams["font.size"] = 12

    if speedup:
        ax.set_ylabel("Speedup")
    else:
        ax.set_ylabel("Time (seconds)")
    if grid:
        ax.set_xlabel("Number of grid points")
    else:
        ax.set_xlabel("Number of particles")

    ax.legend(handles=fake_lines)
    ax.set_xscale('log')
    ax.set_yscale('log')
    #a1.set_ylim([0.1, 5])

    plt.show()
    #plt.savefig(outname+".png", dpi=300)
