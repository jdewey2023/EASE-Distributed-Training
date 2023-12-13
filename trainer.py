import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ndcg_score, f1_score
import numpy as np
import pandas as pd	
import torch

# EASE model copied due to issues with Slurm
class EASE:
	def __init__(self, df, implicit=False, reg=0.05):
		self.reg = reg
		self.indices = torch.LongTensor(df[["user_id", "item_id"]].values)
		if implicit:
			self.values = torch.ones(self.indices.shape[0])
		else:
		self.values = torch.FloatTensor(df["rating"].to_numpy())
		self.sparse = torch.sparse.FloatTensor(self.indices.t(), self.values)

	def fit(self):
		# Sparse matrix multiplication (G = X^T * X)
		G = torch.sparse.mm(self.sparse.t(), self.sparse)
		# Since the regularization term and inverse operation require dense matrices,
		# we selectively convert the relevant parts to dense.
		G_dense = G.to_dense()
		G_dense += torch.eye(G_dense.shape[0]) * self.reg
		# Inverse operation (still requires dense matrix)
		P = G_dense.inverse()
		# Remaining operations
		B = P / (-1 * P.diag())
		B = B + torch.eye(B.shape[0])
		self.B = B
		return

	def predict(self, pred_df, k=5, remove_owned=True):
		unique_users = torch.LongTensor(pred_df["user_id"].unique())
		user_tensor = self.sparse.to_dense().index_select(dim=0, index=unique_users)
		preds_tensor = user_tensor @ self.B - user_tensor * remove_owned
		top_k_indices = preds_tensor.topk(k, dim=1).indices
		pred_items = [indices.numpy() for indices in top_k_indices]
		return pd.DataFrame({'user_id': unique_users, 'predicted_items': pred_items})

# Get dataframe from a csv file
def get_data(i):
	return pd.read_csv(f'data/data_{i}.d')		

def main():
	# DDP
	num_data_files = 20
	n = 0
	s = 1
	if 'WORLDSIZE' in os.environ:
		s = int(os.environ["WORLDSIZE"])
	if 'SLURM_PROCID' in os.environ:
		n = int(os.environ["SLURM_PROCID"])
	print(f'Proc: {n}')
	user_encoder = LabelEncoder()
	item_encoder = LabelEncoder()
	df = pd.DataFrame()
	for i in range(n+1, num_data_files+1, s):
		df = pd.concat([df, get_data(i)])
	df["user_id"] = user_encoder.fit_transform(df["user_id"])
	df["item_id"] = item_encoder.fit_transform(df["item_id"])
	train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
	model = EASE(train_df, 10)
	model.fit()
	torch.save(model.B, f'log/tensor_{n}')
	

if __name__ == '__main__':
	main()