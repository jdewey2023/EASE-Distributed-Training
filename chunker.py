import json
from math import ceil
import pandas as pd

data = []
df = pd.DataFrame()
num = 1
n_chunks = 20

# Output chunk of dataset to a csv file
def output():
	global num
	chunks = []
	chunk_size = ceil(len(df) / n_chunks)
	for i in range(n_chunks):
		chunks.append(split_set(df, i, chunk_size))
	for i in range(num, num + n_chunks):
		with open(f'data_{i}.d', 'w') as file:
			file.write(chunks[i%num].to_csv())
	num += n_chunks

# Convert list of json objects to dataframe
def convert_data():
	global df
	df = pd.DataFrame.from_records(data)
	df = df[['reviewerID', 'asin', 'overall']].rename(columns={'reviewerID': 'user_id', 'asin': 'item_id', 'overall': 'rating'})

# Read a file and extract json
def read(fp):
	global data
	for line in fp:
		data.append(json.loads(line))

# Extract a chunk from a datafram
def split_set(df, i, chunk_size):
	return df[i * chunk_size : (i+1) * chunk_size]


def main():
	read(open('../data/Books_5.json', 'r'))
	read(open('../data/Kindle_Store_5.json', 'r'))
	convert_data()
	output()

if __name__ == '__main__':
	main()