import os
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

# Copy of metric module due to issues with Slurm
def calculate_metrics(predictions, ground_truth, k):
	precision_at_k = []
	recall_at_k = []
	ndcg_at_k = []
	for user_id in predictions['user_id']:
		pred_items = set(predictions[predictions['user_id'] == user_id]['predicted_items'].iloc[0][:k])
		if user_id in ground_truth['user_id'].values:
			actual_items_tuple = tuple(ground_truth[ground_truth['user_id'] == user_id]['actual_items'].iloc[0])
			actual_items = set(actual_items_tuple)
		else:
			actual_items = set()
		num_relevant_items = len(actual_items.intersection(pred_items))
		precision = num_relevant_items / len(pred_items) if pred_items else 0
		recall = num_relevant_items / len(actual_items) if actual_items else 0
		precision_at_k.append(precision)
		recall_at_k.append(recall)
		relevance = [1 if item in actual_items else 0 for item in pred_items]
		ndcg_at_k.append(ndcg_score([relevance], [np.ones(len(relevance))]) if relevance else 0)
		avg_precision = np.mean(precision_at_k) if precision_at_k else 0
		avg_recall = np.mean(recall_at_k) if recall_at_k else 0
		avg_ndcg = np.mean(ndcg_at_k) if ndcg_at_k else 0
		avg_f1 = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0
	return avg_precision, avg_recall, avg_ndcg, avg_f1

# Get a dataframe from a csv file
def get_data(i):
	return pd.read_csv(f'data/data_{i}.d')

def main():
	num_data_files = 20
	n = 0
	s = 1
	if 'WORLDSIZE' in os.environ:
		s = int(os.environ["WORLDSIZE"])
	if 'SLURM_PROCID' in os.environ:
		n = int(os.environ["SLURM_PROCID"])
	user_encoder = LabelEncoder()
	item_encoder = LabelEncoder()
	df = pd.DataFrame()
	for i in range(n+1, num_data_files+1, s):
		df = pd.concat([df, get_data(i)])
	df["user_id"] = user_encoder.fit_transform(df["user_id"])
	df["item_id"] = item_encoder.fit_transform(df["item_id"])
	model = EASE(df, 10)
	model.B = torch.load('log/overall')
	predictions = model.predict(df, 10, False)
	predictions['user_id'] = user_encoder.inverse_transform(predictions['user_id'])
	predictions['predicted_items'] = predictions['predicted_items'].apply(lambda x: item_encoder.inverse_transform(x))
	ground_truth = df.groupby('user_id')['item_id'].apply(list).reset_index()
	ground_truth.rename(columns={'item_id': 'actual_items'}, inplace=True)
	ground_truth['user_id'] = user_encoder.inverse_transform(ground_truth['user_id'])
	ground_truth['actual_items'] = ground_truth['actual_items'].apply(lambda x: item_encoder.inverse_transform(x))
	k = 10
	precision, recall, ndcg, f1 = calculate_metrics(predictions, ground_truth, k)
	print(f"Precision@{k}: {precision}")
	print(f"Recall@{k}: {recall}")
	print(f"NDCG@{k}: {ndcg}")
	print(f"F1@{k}: {f1}")


if __name__ == '__main__':
	main()