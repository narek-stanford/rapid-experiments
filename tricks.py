
import h5py
import numpy as np
from json_tricks.np import dump


with h5py.File("model_weights.h5") as hf:
	dataset_names = ('/layer_0/param_0', '/layer_0/param_1')

	weights = {name: np.array(hf.get(name)) for name in dataset_names}
	
	fp = open("weights.json", 'wt')
	dump(weights, fp, indent=4)
	fp.close()

    
