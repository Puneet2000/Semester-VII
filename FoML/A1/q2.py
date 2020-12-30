import pandas as pd 
import numpy as np 
import math

# np.random.seed(1)

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


df = pd.read_csv('./horsekicks.csv')
a = 3.0 #alpha beta values for Gamma
b = 2.0
for corps_idx in range(14):
	# print('Corp : ',df.iloc[corps_idx,0])
	y_train = np.asarray(df.iloc[corps_idx,1:14], dtype=np.uint32)
	y_test = np.asarray(df.iloc[corps_idx,14:-1], dtype=np.uint32)

	lam_mle = np.mean(y_train) # ML estimate
	lam_map = (np.sum(y_train) + a - 1.0)/(len(y_train) + b) #MAP estimate

	print('Poisson MLE lambda : ', lam_mle)
	print('RMSE : ', rmse(lam_mle, y_test))
	print('Poisson MAP lambda : ', lam_map)
	print('RMSE : ', rmse(lam_map, y_test))
	# print(df.iloc[corps_idx,0], '&', lam_mle, '&', rmse(lam_mle, y_test), '&', lam_map, '&', rmse(lam_map, y_test), '\\\\')
	print()


