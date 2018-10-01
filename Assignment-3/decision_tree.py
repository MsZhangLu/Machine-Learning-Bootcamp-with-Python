import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()
		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls

		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the dim of feature to be splitted

		self.feature_uniq_split = None # the feature to be splitted


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of corresponding training samples
					  e.g.  xxxx++++
					  	 _____|_______			
					  	 xx        	xx
					  	 ++++
					branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################

			branches = np.array(branches)

			total_v = np.sum(branches, axis = 0)

			total = total_v / np.sum(total_v)

			r = np.divide(m, total_v)

			r = np.sum(np.where(r != 0, (-1) * r * np.log2(r), r), axis = 0)

			entropy = np.sum(r * total_v / np.sum(total_v))

			return entropy

		
		best_split_entropy = np.iinfo(int).max
		
		if len(self.features[0]) < 1:
			self.splittable = False
			return

		for idx_dim in range(len(self.features[0])):
		# ############################################################
		# # TODO: compare each split using conditional entropy
		# #       find the best split
		# ############################################################
			f = np.array(self.features)

			B = np.unique(f[:, idx_dim])

			m = np.full((self.num_cls, len(B)), 0)
			
			C = np.unique(np.array(self.labels))

			for j, c in zip(range(len(C)), C):
				for i, b in zip(range(len(B)), B):
					for k in range(len(self.features)):
						if self.features[k][idx_dim] == b and self.labels[k] == c:
							m[j][i] += 1

			if idx_dim == 0:
				self.dim_split = 0
				self.feature_uniq_split = list(B)
				if conditional_entropy(m) < best_split_entropy: 
					best_split_entropy = conditional_entropy(m)

			else:
				if conditional_entropy(m) < best_split_entropy: 
					self.dim_split = idx_dim
					self.feature_uniq_split = list(B)
					best_split_entropy = conditional_entropy(m)
					best_children = m

		# ############################################################
		# # TODO: split the node, add child nodes
		# ############################################################

		branches_name = self.feature_uniq_split
		selected_idx = {key: [] for key in branches_name}

		for i in range(len(self.features)):
			selected_idx[self.features[i][self.dim_split]].append(i)

		for si in selected_idx:
			new_features = []
			new_label = []
			for i in selected_idx[si]:
				new_features.append(self.features[i][:self.dim_split] + self.features[i][self.dim_split+1:] )
				new_label.append(self.labels[i])
			new_num_cls = np.max(self.labels)+1
			self.children.append(TreeNode(new_features, new_label, new_num_cls))

		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			feature = feature[:self.dim_split] + feature[self.dim_split+1:] 
			return self.children[idx_child].predict(feature)
		else:
			return self.cls_max



