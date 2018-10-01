import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		########################################################
		# TODO: implement "predict"
		########################################################
		result = []		
		preds = np.dot(np.array(self.betas), np.array([ele.predict(features) for ele in self.clfs_picked]))
		for pred in preds:
			pred_sign = 1 if pred >= 0 else -1
			result.append(pred_sign)
		return result

		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################

		w = np.full(len(labels), 1/len(labels))

		for iter in range(self.T):
			ht = None
			err = np.iinfo(np.int32).max
			for clfs in self.clfs:
				pred = np.array(clfs.predict(features))
				y = np.array(labels)
				ind = np.abs(0.5 *(pred - y))

				e = np.sum(np.not_equal(pred, y) * w)

				if e < err:
					err = e
					ht = clfs

			self.clfs_picked.append(ht)

			beta = 0.5 * np.log((1-err)/err)

			self.betas.append(beta)

			pred = np.array(ht.predict(features))

			for n in range(len(w)):
				if y[n] == pred[n]:
					w[n] = w[n]*np.exp(-beta)
				else:
					w[n] = w[n]*np.exp(beta)

			w_sum = sum(w)

			w = w / w_sum
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)


class LogitBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "LogitBoost"
		return

	def train(self, features: List[List[float]], labels: List[int]):
		############################################################
		# TODO: implement "train"
		############################################################

		pi = np.full(len(features), 1/2)
		f = np.full(len(features), 0)

		for iter in range(self.T):
			y = np.array(labels)
			z = (0.5*(y+1) - pi) / (pi * (1 - pi))

			w = pi * (1-pi)

			ht = None
			err = np.iinfo(np.int32).max

			for clfs in self.clfs:
				pred = clfs.predict(features)

				e = sum(w*((z - pred)**2))
				if e <= err:
					err = e
					ht = clfs
			
			self.clfs_picked.append(ht)

			f = f + 0.5 * np.array(ht.predict(features))
			self.betas.append(1/2)
			pi = 1 / (1 + np.exp(-2*f))
		
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)
