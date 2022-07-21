import sys, numpy as np
from  matplotlib import pyplot as plt

from Cabana_PlotBenchmarkUtils import *

colors = ["#E31A1C", "#4291C7"]
categories = data.getAllCategories()
print(categories)
for backend in data.getAllBackends():
    for cat, c in zip(["gather"], colors[1:2]):
        print(cat, c)
        for n in data.getAllCommFractions():
            prev = DataDescription(backend, "halo", cat, ["slice"])
            prev_results = AllSizesSingleResult(data, prev, n)

            new = DataDescription(backend, "halo", cat, ["buffer","slice"])
            new_results = AllSizesSingleResult(data, new, n)

            x = np.array(prev_results.sizes)
            y = np.array(prev_results.data)# / np.array(new_results.data)
            plotResults(fig1, ax1, x, y, backend, c)

min_max = data.minMaxSize()
createPlot(fig1, ax1, min_max, speedup=True)
plt.show()
#plt.savefig(outname+".png", dpi=300)
