from util import xor
from sklearn.utils import shuffle
import numpy as np, matplotlib.pyplot as plt

class neural(object):

	def forward(self,X,Y,w,b1,v,b2):

		a = X.dot(w) + b1
		Z = np.tanh(a)
		A = Z.dot(v) + b2
		Y = np.exp(A)/np.exp(A).sum(axis=1,keepdims=True)
		P = np.argmax(Y,axis=1)
		return Y,P,Z

	def derivative_v(self,T,Y,Z):

		return Z.T.dot(T - Y)

	def derivative_b2(self,T,Y):

		return (T - Y).sum(axis=0)

	def derivative_w(self,X,T,Y,v,Z):

		a = (T - Y).dot(v.T)
		b = (1 - Z * Z)
		return X.T.dot(a * b)

	def derivative_b1(self,T,Y,v,Z):

		a = (T - Y).dot(v.T)
		b = (1 - Z * Z)
		return (a * b).sum(axis=0)

	def cross_entropy(self,T,Y):

		e = T * np.log(Y) + (1 - T) * np.log(1 - Y)
		return -np.sum(e)

	def classification_rate(self,Y,P):

		return np.mean(Y == P)

if __name__ == '__main__':

	X,Y = xor()
	X,Y = shuffle(X,Y)
	T = np.zeros((len(Y),len(set(Y))))
	idx_row = np.arange(len(Y))
	idx_col = Y.astype(int)
	T[idx_row,idx_col] = 1
	N = len(Y)//2
	Xtrain = X[:N]
	Ytrain = Y[:N]
	T_train = T[:N]
	Xtest = X[N:]
	Ytest = Y[N:]
	T_test = T[N:]
	D = X.shape[1]
	M = 5
	K = len(set(Y))
	w = np.random.randn(D,M)
	b_1 = np.random.randn(M)
	v = np.random.randn(M,K)
	b_2 = np.random.randn(K)
	learning_rate = 10e-5
	model = neural()
	train_costs = []
	test_costs = []
	for i in range(5000):
			Y_train,P_train,Z_train = model.forward(Xtrain,Ytrain,w,b_1,v,b_2)
			Y_test,P_test,Z_test = model.forward(Xtest,Ytest,w,b_1,v,b_2)
			c_train = model.cross_entropy(T_train,Y_train)
			c_test = model.cross_entropy(T_test,Y_test)
			train_costs.append(c_train)
			test_costs.append(c_test)
			v += learning_rate*model.derivative_v(T_train,Y_train,Z_train)
			b_2 += learning_rate*model.derivative_b2(T_train,Y_train)
			w += learning_rate*model.derivative_w(Xtrain,T_train,Y_train,v,Z_train)
			b_1 += learning_rate*model.derivative_b1(T_train,Y_train,v,Z_train)
			if (i+1) % 100 == 0:
				print(i+1,c_train,c_test)
	legend_1 = plt.plot(train_costs,label='Train')
	legend_2 = plt.plot(test_costs,label='Test')
	plt.legend([legend_1,legend_2])
	plt.show()
	print('Train Classification Rate: ',model.classification_rate(Ytrain,P_train))
	print('Test Classification Rate: ',model.classification_rate(Ytest,P_test))