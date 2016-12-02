# coding: utf-8
from __future__ import print_function
import seaborn as sn
import matplotlib.pyplot as plt
import glob


def get_epochNo_to_lossVal(weights_hdf5_filenames):
	dd = {int(fname[fname.find('.')+1:fname.find('-')]): float(fname[fname.find('-')+1:fname.rfind('.')]) for fname in weights_hdf5_filenames}
	print(dd)
	return dd

lossRecords = glob.glob("*.hdf5")
en2lv = get_epochNo_to_lossVal(lossRecords)

#plt.plot(en2lv.keys(), en2lv.values())

