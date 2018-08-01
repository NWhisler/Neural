from process import encode
from sklearn.utils import shuffle
import pandas as pd, numpy as np, matplotlib.pyplot as plt 

class Neural(object):

	def __init__(self,hidden_layers,learning_rate,iterations,activation):

		self.layers = hidden_layers
		self.learning_rate = learning_rate
		self.iterations = iterations
		self.activation = activation

	def activations(self,a):

		if self.activation == 'sigmoid':
			return 1/(1 + np.exp(-a))
		elif self.activation == 'tanh':
			return np.tanh(a)
		else:
			return a>0

	def cross_entropy(self):

		return -np.mean(self.T * np.log(self.Y))

	def fit(self,X,Y):

		N = len(Y)
		D = X.shape[1]
		M = self.layers[1]
		K = len(set(Y))
		self.Classifications = Y
		self.T = np.zeros((N,K))
		idx_row = np.arange(N)
		idx_col = Y.astype(int)
		self.T[idx_row,idx_col] = 1
		layers = 1 if len(self.layers) == 1 else self.layers[0]
		Z = [0] * layers
		v = [np.random.randn(self.layers[1],self.layers[1]) for i in range(layers - 1)]
		v.append(np.random.randn(M,K))
		b = [np.random.randn(self.layers[1]) for i in range(layers - 1)]
		b.append(np.random.randn(K))
		w = np.random.randn(D,M)
		b_0 = np.random.random(M)
		self.cr = []
		self.cost = []
		for gamma in range(self.iterations):
			Z[0] = X.dot(w) + b[0]
			for i in range(layers - 1):
				Z[i+1] = self.activations(Z[i].dot(v[i]) + b[i])
			A = Z[-1].dot(v[-1]) + b[-1]
			self.Y = np.exp(A)/np.exp(A).sum(axis=1,keepdims=True)
			self.P = np.argmax(self.Y,axis=1)
			self.cost.append(self.cross_entropy())
			self.cr.append(self.score())
			Z = np.array(Z)
			dv = [0] * layers
			db = [0] * layers
			if self.activation == 'sigmoid':
				dz = Z * (1 - Z)
			elif self.activation == 'tanh':
				dz = (1 - Z * Z)
			else:
				dz = Z>0
			a = (self.T - self.Y)
			dv[-1] = Z[-1].T.dot(a)
			idx_sorted = sorted(list(range(len(dv))),reverse=True)
			idx_sorted = idx_sorted[:-1]
			idx = []
			for i in idx_sorted:
				idx.append(i)
				idx.append(i)
			for i in idx_sorted:
				count = 0
				for e in idx:
					if e == (i-1):
						break
					else:
						if count % 2 == 0:
							a = a.dot(v[i].T)
						else:
							a = a * dz[i]
						count += 1
				if i == 1:
					dw = X.T.dot((a.dot(v[i-1].T))*dz[i-1] )
				dv[i-1] = Z[i-1].T.dot(a)
			a = (self.T - self.Y)
			db[-1] = a.sum(axis=0)
			for i in idx_sorted:
				count = 0
				for e in idx:
					if e == (i-1):
						break
					else:
						if count % 2 == 0:
							a = a.dot(v[i].T)
						else:
							a = a * dz[i]
						count += 1
				if i == 1:
					db_0 = ((a.dot(v[i-1].T))*dz[i-1]).sum(axis=0) 
				db[i-1] = a.sum(axis=0)
			for i in range(len(v)):
				v[i] += self.learning_rate*dv[i]
			for i in range(len(b)):
				b[i] += self.learning_rate*db[i]
			w += self.learning_rate * dw
			b_0 += self.learning_rate * db_0
			if gamma % 100 == 0:
				print(gamma,self.cost[gamma])

	def graph_cost(self):

		plt.plot(self.cost)
		plt.title('Cost')
		plt.show()

	def graph_cr(self):

		plt.plot(self.cr)
		plt.title('Classification Rate')
		plt.show()

	def project(self):

		return self.P

	def score(self):

		return np.mean(self.Classifications == self.P)

if __name__ == '__main__':

	X,Y = encode()
	X,Y = shuffle(X,Y)
	model = Neural((2,100),50e-7,10000,'relu')
	model.fit(X,Y)
	model.graph_cost()
	model.graph_cr()
	print('Final Classification Rate: ',model.score())
	best_cr =  0
	iteration = 0
	for e,i in enumerate(model.cr):
		if i > best_cr:
			best_cr = i
			iteration = e
	print('Best Classification Rate: ',best_cr)
	print('Iteration: ',iteration)