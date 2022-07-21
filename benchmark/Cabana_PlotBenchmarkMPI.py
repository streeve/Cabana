import sys, numpy as np
from  matplotlib import pyplot as plt

from Cabana_BenchmarkPlotUtils import *

#outname = sys.argv[1]
filelist = sys.argv[1:]
print(filelist)
data = AllDataMPI(filelist)

fig1 = plt.figure()
ax1 = fig1.add_subplot(111)

color_list = ["#E31A1C", "#4291C7", "g", "y"]
color_dict = {}
categories = data.getAllCategories()
for c, cat in enumerate(categories):
    color_dict[cat] = color_list[c]

for backend in data.getAllBackends()[0:1]:
    for cat in categories:
        for type in data.getAllTypes()[1:2]:
            for param in [["fast"], ["aosoa"]]: #data.getAllParams():
                for n in data.getAllCommFractions():
                    desc = DataDescription(backend, type, cat, param)
                    result = AllSizesSingleResult(data, desc, n)

                    x = np.array(result.sizes)
                    y = np.array(result.data)
                    plotResults(fig1, ax1, x, y, backend, color_dict[cat])

min_max = data.minMaxSize()
createPlot(fig1, ax1, min_max)
