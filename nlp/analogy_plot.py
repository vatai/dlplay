#!/bin/env python

import sys
import vecto.benchmarks.visualize; 
from matplotlib import pyplot as plt; 

name=sys.argv[1]
vecto.benchmarks.visualize.plot_accuracy(name); 
plt.savefig(name+".pdf", bbox_inches="tight")

