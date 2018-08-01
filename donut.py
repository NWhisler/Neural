from util import donut
from sklearn.utils import shuffle
import numpy as np,matplotlib.pyplot as plt 

class neural(object):

	def __init__(self,X,Y):

		self.X = X
		self.classifications = Y
		self.N,self.D = X.shape
		self.M = 5
		self.K = len(set(Y))
		self.T = np.zeros((self.N,self.K))
		idx_row = np.arange(len(Y))
		idx_col = Y.astype(int)
		self.T[idx_row,idx_col] = 1

	def sigmoid(self,a):

		return 1/(1 + np.exp(-a))

	def forward(self,w,v1,v2,b0,b1,b2):

		a_0 = self.X.dot(w)+ b0
		Z_1 = self.sigmoid(a_0)
		a_1 = Z_1.dot(v1) + b1
		Z_2 = self.sigmoid(a_1)
		A = Z_2.dot(v2) + b2
		self.Y = np.exp(A)/np.exp(A).sum(axis=1,keepdims=True)
		self.P = np.argmax(self.Y,axis=1)
		return Z_1,Z_2

	def derivative_v2(self,Z_2):

		return Z_2.T.dot(self.T - self.Y)

	def derivative_b2(self):

		return (self.T - self.Y).sum(axis=0)

	def derivative_v1(self,v_2,Z_1,Z_2):

		a = (self.T - self.Y).dot(v_2.T)
		b = Z_2 * (1 - Z_2)
		return Z_1.T.dot(a * b)

	def derivative_b1(self,v_2,Z_2):

		a = (self.T - self.Y).dot(v_2.T)
		b = Z_2 * (1 - Z_2)
		return (a * b).sum(axis=0)

	def derivative_w(self,v_1,v_2,Z_1,Z_2):

		a = (self.T - self.Y).dot(v_2.T)
		b = Z_2 * (1 - Z_2)
		c = (a * b).dot(v_1.T)
		d = Z_1 * (1 - Z_1)
		return self.X.T.dot(c * d)

	def derivative_b0(self,v_1,v_2,Z_1,Z_2):

		a = (self.T - self.Y).dot(v_2.T)
		b = Z_2 * (1 - Z_2)
		c = (a * b).dot(v_1.T)
		d = Z_1 * (1 - Z_1)
		return (c * d).sum(axis=0)

	def cross_entropy(self):

		return -np.mean(self.T - np.log(self.Y))

	def classification_rate(self):

		return np.mean(self.classifications == self.P)

if __name__ == '__main__':

	X,Y = donut()
	X,Y = shuffle(X,Y)
	N = len(Y)//2
	Xtrain = X[:N]
	Ytrain = Y[:N]
	Xtest = X[N:]
	Ytest = Y[N:]
	train_model = neural(Xtrain,Ytrain)
	test_model = neural(Xtest,Ytest)
	train_cost = []
	test_cost = []
	D = X.shape[1]
	M = 5
	J = 10
	K = len(set(Y))
	w = np.random.randn(D,M)
	v_1 = np.random.randn(M,J)
	v_2 = np.random.randn(J,K)
	b_0 = np.zeros(M)
	b_1 = np.zeros(J)
	b_2 = np.zeros(K)
	learning_rate = 10e-5
	for i in range(10000):
		#w,v1,v2,b0,b1,b2
		Z1_train,Z2_train = train_model.forward(w,v_1,v_2,b_0,b_1,b_2)
		Z1_test,Z2_test = test_model.forward(w,v_1,v_2,b_0,b_1,b_2)
		c_train = train_model.cross_entropy()
		c_test = test_model.cross_entropy()
		train_cost.append(c_train)
		test_cost.append(c_test)
		v_2 += learning_rate*train_model.derivative_v2(Z2_train)
		b_2 += learning_rate*train_model.derivative_b2()
		v_1 += learning_rate*train_model.derivative_v1(v_2,Z1_train,Z2_train)
		b_1 += learning_rate*train_model.derivative_b1(v_2,Z2_train)
		w += learning_rate*train_model.derivative_w(v_1,v_2,Z1_train,Z2_train)
		b_0 += learning_rate*train_model.derivative_b0(v_1,v_2,Z1_train,Z2_train)
		if (i+1) % 100 == 0:
			print(i+1,c_train,c_test)
	legend1 = plt.plot(train_cost,label='Train')
	legend2 = plt.plot(test_cost,label='Test')
	plt.legend([legend1,legend2])
	plt.show()
	print('Train Classification Rate: ',train_model.classification_rate())
	print('Test Classification Rate: ',test_model.classification_rate())