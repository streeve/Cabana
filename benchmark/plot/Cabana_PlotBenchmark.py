import sys, numpy as np
from  matplotlib import pyplot as plt

from Cabana_BenchmarkPlotUtils import *

def plotAll(fig, ax, data):
    color_dict = getColors(data)
    for backend in data.getAllBackends():
        for cat in data.getAllCategories():
            for type in data.getAllTypes():
                for param in data.getAllParams():
                    desc = ManualDataDescription(backend, type, cat, param)
                    result = AllSizesSingleResult(data, desc)

                    x = np.array(result.sizes)
                    if data.grid: x = x**3
                    y = np.array(result.data)
                    plotResults(fig, ax, x, y, backend, color_dict[cat])

def plotAllMPI(fig, ax, data):
    color_dict = getColors(data)
    for backend in data.getAllBackends()[1:2]:
        for cat in data.getAllCategories()[0:2]:
            for type in data.getAllTypes():
                for param in data.getAllParams():
                    for n in data.getAllCommFractions():
                        print(backend, cat, type, param)
                        desc = ManualDataDescription(backend, type, cat, param)
                        result = AllSizesSingleResultMPI(data, desc, n)

                        x = np.array(result.sizes)
                        if data.grid: x = x**3
                        y = np.array(result.data)
                        plotResults(fig, ax, x, y, backend, color_dict[cat])

def plotCompareHostDevice(fig, ax, data):
    color_dict = getColors(data)
    for backend in data.getAllBackends():
        if backend == "host": continue
        for cat in data.getAllCategories():
            for type in data.getAllTypes():
                for param in data.getAllParams():
                    print(backend, cat, type, param)
                    desc = ManualDataDescription(backend, type, cat, param)
                    result = AllSizesSingleResult(data, desc)
                    desc2 = ManualDataDescription("host", type, cat, param)
                    result2 = AllSizesSingleResult(data, desc2)

                    num_1 = len(result.data)
                    num_2 = len(result2.data)
                    max = num_1 if num_1 < num_2 else num_2

                    x = np.array(result.sizes[:max])
                    if data.grid: x = x**3
                    y = np.array(result2.data[:max]) / np.array(result.data[:max])
                    plotResults(fig1, ax1, x, y, backend, color_dict[cat])

def plotCompareHostDeviceMPI(fig, ax, data):
    color_dict = getColors(data)
    for backend in data.getAllBackends():
        if backend == "hip_host" or backend == "host_host": continue
        for cat in data.getAllCategories()[0:2]:
            for type in data.getAllTypes():
                for param in data.getAllParams():
                    for n in data.getAllCommFractions():
                        print(backend, cat, type, param)
                        desc = ManualDataDescription(backend, type, cat, param)
                        result = AllSizesSingleResultMPI(data, desc, n)
                        desc2 = ManualDataDescription("hip_host", type, cat, param)
                        result2 = AllSizesSingleResultMPI(data, desc2, n)

                        num_1 = len(result.data)
                        num_2 = len(result2.data)
                        max = num_1 if num_1 < num_2 else num_2

                        x = np.array(result.sizes)
                        if data.grid: x = x**3
                        y = np.array(result2.data) / np.array(result.data[:max])
                        plotResults(fig, ax, x, y, backend, color_dict[cat])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit("Provide Cabana benchmark file path(s) on the command line.")
    filelist = sys.argv[1:]
    print(filelist)
    #data = AllData(filelist)
    data = AllDataMPI(filelist)
    #data = AllDataInterpolation(filelist, grid=True)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    #plotAll(fig1, ax1, data)
    #plotCompareHostDevice(fig1, ax1, data)
    #plotAllMPI(fig1, ax1, data)
    plotCompareHostDeviceMPI(fig1, ax1, data)
    createPlot(fig1, ax1, data,
               speedup=True)#, cpu_name="POWER9", gpu_name="V100")
