import numpy as np
import pickle as pk

class Network(object):
	"""An object which contains the nodes in the network."""
	def __init__(self, layers, neurons_per_layer, eta, train_data):
		super(Network, self).__init__()
		self.num_layers = layers
		self.neurons_per_layer = neurons_per_layer
		self.eta = eta 
		self.train_data = train_data
		self.weights = np.random.rand(layers, neurons_per_layer, neurons_per_layer)
		self.biases = np.random.rand(neurons_per_layer, layers)
		# self.weights = np.ones((layers, neurons_per_layer, neurons_per_layer))
		# self.biases = np.ones((neurons_per_layer, layers))
		self.N = np.zeros((len(train_data), self.neurons_per_layer, self.num_layers))
		self.Z = np.zeros((len(train_data), self.neurons_per_layer, self.num_layers + 1))
		self.A = np.zeros((len(train_data), self.neurons_per_layer, self.num_layers + 1))


	def feedforward(self, x, i):
		self.A[i,:,0] = x 
		for l in range(self.num_layers):
			z = np.dot(self.weights[l], self.A[i,:,l]) + self.biases[l]
			self.Z[i,:,l+1] = z
			self.A[i,:,l+1] = self.sigmoid(z)

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def sig_prime(self, z):
		return np.exp(-z) / np.power(1 + np.exp(-z), 2)

	def cross_entropy(self, y, a):
		return -np.dot(y, np.log(a)) + np.dot((1 - y), np.log(1 - a))

	def backprop(self, y, i):
		self.N[i,:,-1] = self.A[i,:,-1] - y
		for l in reversed(range(1, self.num_layers)):
			self.N[i,:,l-1] = np.multiply(np.dot(np.transpose(self.weights[l]), self.N[i,:,l]), self.sig_prime(self.Z[i,:,l]))

	def grad_descent(self):
		for i in range(len(self.train_data)):
			x = self.train_data[i][0]
			y = self.train_data[i][1]
			self.feedforward(x, i)
			self.backprop(y, i)
		for l in range(self.num_layers):
			self.weights[:,l] = self.weights[:,l] - (self.eta / len(self.train_data)) * np.sum(np.dot(self.N[:,:,l], np.transpose(self.A[:,:,l-1])))
			self.biases[:,l] = self.biases[:,l] - (self.eta / len(self.train_data)) * np.sum(self.N[:,:,l])
			print(self.weights)
			print(self.biases)

		




		