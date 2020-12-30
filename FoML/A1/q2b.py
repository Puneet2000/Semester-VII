import pandas as pd 
import numpy as np 
import math

# np.random.seed(1)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


df = pd.read_csv('./horsekicks.csv')
a = 1.0
b = 0.5

for corps_idx in range(14):
	# print('Corp : ',df.iloc[corps_idx,0])
	y_train = np.asarray(df.iloc[corps_idx,1:14], dtype=np.uint32)
	y_test = np.asarray(df.iloc[corps_idx,14:-1], dtype=np.uint32)

	lam = (np.sum(y_train) + a - 1.0)/(13.0 + b)

	print('Poisson lambda : ', lam)
	print('RMSE : ', rmse(lam, y_test))
	print()


