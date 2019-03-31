import cPickle 
import gzip 

import numpy as np

def load_mnist():
	f = gzip.open('./data/mnist.pkl.gz', 'rb')
	tr_d, va_d, te_d = cPickle.load(f)
	training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
	training_results = [label_array(y) for y in tr_d[1]]
	training_data = zip(training_inputs, training_results)
	validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
	validation_data = zip(validation_inputs, va_d[1])
	test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
	test_data = zip(test_inputs, te_d[1])
	return (training_data, validation_data, test_data)

def label_array(y):
	arr = np.zeros((10,1))
	arr[y] = 1.0
	return arr