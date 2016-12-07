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
plt.plot([0,1,2,3], en2lv.values())
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



