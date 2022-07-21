import sys, numpy as np
from  matplotlib import pyplot as plt

from Cabana_BenchmarkPlotUtils import *

colors = ["#E31A1C", "#4291C7"]
categories = data.getAllCategories()
print(categories)
for backend in data.getAllBackends():
    for cat, c in zip(["gather"], colors[1:2]):
        print(cat, c)
        for n in data.getAllCommFractions():
            prev = ManualDataDescription(backend, "halo", cat, ["slice"])
            prev_results = AllSizesSingleResultMPI(data, prev, n)

            new = ManualDataDescription(backend, "halo", cat, ["buffer","slice"])
            new_results = AllSizesSingleResultMPI(data, new, n)

            x = np.array(prev_results.sizes)
            y = np.array(prev_results.data)# / np.array(new_results.data)
            plotResults(fig1, ax1, x, y, backend, c)

min_max = data.minMaxSize()
createPlot(fig1, ax1, min_max, speedup=True)
