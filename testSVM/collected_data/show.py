import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read(path):
	df = pd.read_csv(path, header = None)
	df = df[0: 1]
	data = []
	for i in df:
		data.append(df.iloc[0, i])
	return data


if __name__ == '__main__':
	while True:
		path = input('input file path: ')
		data = read(path)
		plt.plot(data)
		plt.show()