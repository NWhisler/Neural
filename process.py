import numpy as np, pandas as pd

def encode():
	
	df = pd.read_csv('ecommerce_data.csv')
	data = df.as_matrix()
	X_test = X = data[:,:-1]
	X[:,1] = (X[:,1] - X[:,1].mean())/X[:,1].std()
	X[:,2] = (X[:,2] - X[:,2].mean())/X[:,2].std()
	T = data[:,-1]
	one_hot = np.zeros((len(X),4))
	idx_col = list(map(int,X[:,-1]))
	idx_row = list(range(len(X)))
	one_hot[idx_row,idx_col] = 1
	X_add = np.zeros((len(X),3))
	X = np.concatenate((X,X_add),axis=1)
	X[:,-4:] = one_hot
	return X,T