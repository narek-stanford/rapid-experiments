



def preprocess(X, Y):
	# converting RGB -> BGR
	X = X[:,:,:, [2,1,0]]
	
    return X