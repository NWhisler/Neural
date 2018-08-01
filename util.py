import numpy as np, matplotlib.pyplot as plt 

def xor():

	# X = np.array([[0,0],[1,0],[0,1],[1,1]])
	# Y = np.array([0,1,1,0])
	N = 500
	D = 2
	X_1 = np.random.randn(N,D) + np.array([-2,2])
	X_2 = np.random.randn(N,D) + np.array([2,-2])
	X_3 = np.random.randn(N,D) + np.array([-2,-2])
	X_4 = np.random.randn(N,D) + np.array([2,2])
	X = np.vstack((X_1,X_2,X_3,X_4))
	Y = np.array([0] * 2 * N + [1] * 2 * N)
	plt.scatter(X[:,0],X[:,1],c=Y,s=100,alpha=.5)
	plt.show()
	return X,Y

def donut():

	N,D = 1000,2
	R_inner = 5
	theta_inner = np.random.randn(N)
	X_inner = np.vstack((R_inner * np.cos(theta_inner),R_inner * np.sin(theta_inner))).T
	R_outer = 10
	theta_outer = np.random.randn(N)
	X_outer = np.vstack((R_outer * np.cos(theta_outer),R_outer * np.sin(theta_outer))).T
	X = np.vstack((X_inner,X_outer))
	Y = np.array([0] * N + [1] * N)
	plt.scatter(X[:,0],X[:,1],c=Y,s=100,alpha=.5)
	plt.show()
	return X,Y