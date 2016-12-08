# coding: utf-8
from __future__ import print_function
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sn


def get_epochNo_to_lossVal(weights_hdf5_filenames):
	dd = {}
	for fname in weights_hdf5_filenames:
		key = int(fname[fname.find('.')+1:fname.find('-')])
		val = float(fname[fname.find('-')+1:fname.rfind('.')])
		dd[key] = val
	return dd

lossRecords = glob.glob("*.hdf5")
en2lv = get_epochNo_to_lossVal(lossRecords)
print(en2lv)

#df = pd.DataFrame.from_dict(en2lv, 'index')
#df.plot()
plt.plot(range(len(en2lv.keys())), en2lv.values())
plt.show()


def plotFunc(fnames):
	keys = [int(fname[fname.find('.')+1:fname.find('-')]) for fname in fnames]
	values = [float(fname[fname.find('-')+1:fname.rfind('.')]) for fname in fnames]
	plt.plot(keys, values)
plotFunc(lossRecords)
plt.show()


twinLosses = glob.glob("*continued*hdf5")

trLosses = [float(fname[fname.find('-')+1:fname.rfind('-')]) for fname in sorted(twinLosses)]
valLosses = [float(fname[fname.rfind('-')+1:fname.rfind('.')]) for fname in sorted(twinLosses)]
plt.plot([0.2062, 0.2013, 0.2007, 0.1956, 0.1937, 0.1895, 0.1855, 0.1852, 0.1827, 0.1788])
plt.plot([0.2228, 0.2115, 0.2092, 0.2045, 0.2017, 0.2001, 0.2031, 0.1992, 0.1955, 0.1959])
plt.show()


