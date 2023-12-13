import torch

def main():
	n = 5
	tensors = []
	for i in range(n):
		tensors.append(torch.load(f'log/tensor_{i}'))
	result = tensors[0]
	for tensor in tensors[1:]:
		result += tensor
	result /= n
	print(result)
	torch.save(result, 'log/overall')


if __name__ == '__main__':
	main()