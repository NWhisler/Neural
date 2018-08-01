from util import donut, xor
from process import encode
from sklearn.utils import shuffle
import numpy as np, matplotlib.pyplot as plt 

class neural(object):

	def __init__(self,X,Y,layers,layer_dimensionality,functions,learning_rate,alternating=None):

		self.X = X
		self.Classifications = Y
		N = len(Y)
		D = X.shape[1]
		M = layer_dimensionality[0][0]
		K = len(set(Y))
		self.T = np.zeros((N,K))
		idx_row = np.arange(N)
		idx_col = Y.astype(int)
		self.T[idx_row,idx_col] = 1
		self.Z = [0] * layers
		self.w = np.random.randn(D,M)
		self.v = [0] * layers
		for i in range(len(self.v) - 1):
			self.v[i] = np.random.randn(layer_dimensionality[i][0],layer_dimensionality[i][1])
		self.v[-1] = np.random.randn(layer_dimensionality[-1][0],K)
		self.b_0 = np.random.randn(M)
		self.b = [0] * layers
		for i in range(len(self.b) - 1):
			self.b[i] = np.random.randn(layer_dimensionality[i][1])
		self.b[-1] = np.random.randn(K)
		self.functions = functions
		self.learning_rate = learning_rate
		self.alternating = alternating

	def tanh(self,a):

		return np.tanh(a)

	def sigmoid(self,a):

		return 1/(1 + np.exp(-a))

	def forward(self):

		# if len(self.Z) % 2 == 0:
		# 	idx_even_set_even = idx_even[1:]
		# 	idx_odd_set_even = idx_odd[:-1]
		# 	idx_even_set_odd = idx_even
		# 	idx_odd_set_odd = idx_odd
		# else:
		# 	idx_even_set_even = idx_even[1:]
		# 	idx_odd_set_even = idx_odd
		# 	idx_even_set_odd = idx_even[:-1]
		# 	idx_odd_set_odd = idx_odd
		# if self.alternating:
		# 	idx = np.arange(len(self.Z))
		# 	idx_even = np.nonzero(idx % 2 == 0)[0]
		# 	idx_odd = np.nonzero(idx % 2 != 0)[0]
		# 	if self.functions[0] == 'sigmoid':
		# 		self.Z[0] = self.simoid(self.X.dot(self.w) + self.b[0])
		# 		self.Z[idx_even_set_even] = self.sigmoid(self.Z[idx_odd_set_even])
		# 		self.Z[idx_odd_set_odd] = self.tanh(self.Z[idx_even_set_odd])
		# 	else:
		# 		self.Z[0] = self.tanh(self.X.dot(self.w) + self.b[0])
		# 		self.Z[idx_even_set_even] = self.tanh(self.Z[idx_odd_set_even])
		# 		self.Z[idx_odd_set_odd] = self.sigmoid(self.Z[idx_even_set_odd])
		# else:
			
		# 	if self.functions[0] == 'sigmoid':
		# 		self.Z[0] = self.simoid(self.X.dot(self.w) + self.b[0])
		# 		self.Z[idx_even_set_even] = self.sigmoid(self.Z[idx_odd_set_even])
		# 		self.Z[idx_odd_set_odd] = self.sigmoid(self.Z[idx_even_set_odd])
		# 	else:
		# 		self.Z[0] = self.tanh(self.X.dot(self.w) + self.b[0])
		# 		self.Z[idx_even_set_even] = self.tanh(self.Z[idx_odd_set_even])
		# 		self.Z[idx_odd_set_odd] = self.tanh(self.Z[idx_even_set_odd])
		idx = np.arange(len(self.Z) - 1)
		if self.alternating:
			idx_even = np.nonzero(idx % 2 == 0)[0]
			idx_even = idx_even[1:]
			idx_odd = np.nonzero(idx % 2 != 0)[0]
			if self.functions[0] == 'sigmoid':
				self.Z[0] = self.sigmoid(self.X.dot(w) + self.b[0])
				for i in idx_even:
					self.Z[i+1] = self.sigmoid(self.Z[i].dot(self.v[i]) + self.b[i])
				for i in idx_odd:
					self.Z[i+1] = self.tanh(self.Z[i].dot(self.v[i]) + self.b[i])
			else:
				self.Z[0] = self.tanh(self.X.dot(w) + self.b[0])
				for i in idx_even:
					self.Z[i+1] = self.tanh(self.Z[i].dot(self.v[i]) + self.b[i])
				for i in idx_odd:
					self.Z[i+1] = self.sigmoid(self.Z[i].dot(self.v[i]) + self.b[i])
		else:
			if self.functions == 'sigmoid':
				self.Z[0] = self.sigmoid(self.X.dot(self.w) + self.b_0)
				for i in idx:
					self.Z[i+1] = self.sigmoid(self.Z[i].dot(self.v[i]) + self.b[i])
			else:
				self.Z[0] = self.tanh(self.X.dot(self.w) + self.b_0)
				for i in idx:
					self.Z[i+1] = self.tanh(self.Z[i].dot(self.v[i]) + self.b[i])
		A = self.Z[-1].dot(self.v[-1]) + self.b[-1]
		self.Y = np.exp(A)/np.exp(A).sum(axis=1,keepdims=True)
		self.P = np.argmax(self.Y,axis=1)

	def derivative(self):

		dy = (self.T - self.Y).dot(self.v[-1].T)
		if self.functions == 'sigmoid':
			dv = [0] * len(self.v)
			dz = [0] * len(self.Z)
			db = [0] * len(self.b)
			for i in range(len(self.Z)):
				dz[i] = self.Z[i] * (1 - self.Z[i])	
			dv[-1] = self.Z[-1].T.dot(self.T - self.Y)
			idx_sort = sorted(list(range(len(self.v))),reverse=True)
			idx_sort = idx_sort[:-1]
			idx = []
			for i in idx_sort:
				idx.append(i)
				idx.append(i)
			for i in idx_sort:
				count = 0
				a = (self.T - self.Y)
				for e in idx:
					if e == (i-1):
						break
					if count % 2 == 0:
						a = a.dot(self.v[e].T)
					else:
						a = a * dz[e]
					count += 1
				dv[i-1] = self.Z[i-1].T.dot(a)
			v = []
			for i in range(len(self.v)):
				v.append(self.v[i] + self.learning_rate*dv[i])
			self.v = v
			a = dy * dz[-1]
			idx_sort = sorted(list(range(len(self.Z) - 1)),reverse=True)
			idx = []
			for i in idx_sort:
				idx.append(i)
				idx.append(i)
			idx.pop()
			count = 0
			for i in idx:
				if count % 2 == 0:
					a = a.dot(self.v[i].T)
				else:
					a = a * dz[i]
				count += 1
			self.w += self.learning_rate*self.X.T.dot(a)
			idx_sort = sorted(list(range(len(self.b))),reverse=True)
			idx_sort = idx_sort[:-1]
			idx = []
			for i in idx_sort:
				idx.append(i)
				idx.append(i)
			for i in idx_sort:
				count = 0
				a = (self.T - self.Y)
				for e in idx:
					if e == (i-1):
						break
					if count % 2 == 0:
						a = a.dot(self.v[e].T)
					else:
						a = a * dz[e]
					count += 1
				db[i-1] = a.sum(axis=0)
			b = []
			for i in range(len(self.b)):
				b.append(self.b[i] + self.learning_rate*db[i])
			self.b = b
		if self.functions == 'tanh':
			dv = [0] * len(self.v)
			dz = [0] * len(self.Z)
			db = [0] * len(self.b)
			for i in range(len(self.Z)):
				dz[i] = (1 - self.Z[i] * self.Z[i])	
			dv[-1] = self.Z[-1].T.dot(self.T - self.Y)
			idx_sort = sorted(list(range(len(self.v))),reverse=True)
			idx_sort = idx_sort[:-1]
			idx = []
			for i in idx_sort:
				idx.append(i)
				idx.append(i)
			for i in idx_sort:
				count = 0
				a = (self.T - self.Y)
				for e in idx:
					if e == (i-1):
						break
					if count % 2 == 0:
						a = a.dot(self.v[e].T)
					else:
						a = a * dz[e]
					count += 1
				dv[i-1] = self.Z[i-1].T.dot(a)
			v = []
			for i in range(len(self.v)):
				v.append(self.v[i] + self.learning_rate*dv[i])
			self.v = v
			a = dy * dz[-1]
			idx_sort = sorted(list(range(len(self.Z) - 1)),reverse=True)
			idx = []
			for i in idx_sort:
				idx.append(i)
				idx.append(i)
			idx.pop()
			count = 0
			for i in idx:
				if count % 2 == 0:
					a = a.dot(self.v[i].T)
				else:
					a = a * dz[i]
				count += 1
			self.w += self.learning_rate*self.X.T.dot(a)
			idx_sort = sorted(list(range(len(self.b))),reverse=True)
			idx_sort = idx_sort[:-1]
			idx = []
			for i in idx_sort:
				idx.append(i)
				idx.append(i)
			for i in idx_sort:
				count = 0
				a = (self.T - self.Y)
				for e in idx:
					if e == (i-1):
						break
					if count % 2 == 0:
						a = a.dot(self.v[e].T)
					else:
						a = a * dz[e]
					count += 1
				db[i-1] = a.sum(axis=0)
			b = []
			for i in range(len(self.b)):
				b.append(self.b[i] + self.learning_rate*db[i])
			self.b = b	

	def cross_entropy(self):

		return -np.mean(self.T * np.log(self.Y))

	def classification_rate(self):

		return np.mean(self.Classifications == self.P)

if __name__ == '__main__':

	X,Y = encode()
	X,Y = shuffle(X,Y)
	model = neural(X,Y,3,[(100,10),(10,7),(7,4)],'tanh',10e-4)
	costs = []
	cr = []
	for i in range(10000):
		model.forward()
		costs.append(model.cross_entropy())
		model.derivative()
		current_cr = (i+1,model.classification_rate())
		cr.append(current_cr)
		if (i+1) % 100 == 0:
			print(i+1,costs[i])
	plt.plot(costs)
	plt.show()
	iteration = 0
	best_cr = 0
	for i in cr:
		if i[1] > best_cr:
			best_cr = i[1]
			iteration = i[0]
	print('Final Classification Rate: ',model.classification_rate())
	print('Best Classification Rate:',best_cr,'Iteration:',iteration)