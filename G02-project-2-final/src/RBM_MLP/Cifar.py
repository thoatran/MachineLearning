#!/usr/bin/python
import os
import gzip
import pickle
import numpy as np
import gzip
def cifar_encode_one_hot_label(idx):
	enc = np.zeros( 10 )
	enc[ idx ] = 1.0
	return enc

def cifar_decode_one_hot_label(enc):
	return np.argmax( enc )

def cifar_get_accuracy(labels, guesses):
	return np.mean( np.equal( np.argmax( labels, axis = 1 ), np.argmax( guesses, axis = 1 ) ).astype( np.float64 ) )

class Cifar:
	def __init__(self, threshold = True):
		with open('cifar-10-python.tar', 'rb') as fh:
			try:
				training_data, validation_data, testing_data = pickle.load(fh, encoding='latin1' )
			except TypeError:
				training_data, validation_data, testing_data = fpickle.load(fh)

		self.training_digits, self.training_labels = self.format_dataset( training_data, threshold )
		self.validation_digits, self.validation_labels = self.format_dataset( validation_data, threshold )
		self.testing_digits, self.testing_labels = self.format_dataset( testing_data, threshold )

	def getTrainingData(self, count = 0):
		if count == 0:
			return ( self.training_digits, self.training_labels )
		else:
			return self.get_batch( count, self.training_digits, self.training_labels )

	def getValidationData(self, count = 0):
		if count == 0:
			return ( self.validation_digits, self.validation_labels )
		else:
			return self.get_batch( count, self.validation_digits, self.validation_labels )

	def getTestingData(self, count = 0):
		if count == 0:
			return ( self.testing_digits, self.testing_labels )
		else:
			return self.get_batch( count, self.testing_digits, self.testing_labels )

	@staticmethod
	def get_batch(count,digits,labels):
		total = len( digits )
		count = min( count, total )
		idxs  = np.random.choice( np.arange( total ), count, replace=False )
		return ( digits[ idxs ], labels[ idxs ] )

	@staticmethod
	def format_dataset(dataset, threshold):
		digits = np.array( [ np.reshape( x, 3072 ) for x in dataset[ 0 ] ] )
		labels = np.array( [ cifar_encode_one_hot_label( y ) for y in dataset[ 1 ] ] )
		return ( ( digits > 0 ).astype( np.float ) if threshold else digits, labels )
