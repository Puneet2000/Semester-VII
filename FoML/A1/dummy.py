import pandas as pd 
import numpy as np 
import math
from scipy.optimize import minimize

# np.random.seed(1)

def get_objective(x_train, y_train):

	def objective_fn(w):
		y_ = np.dot(x_train,w)
		fact = np.asarray([math.factorial(y) for y in y_train], dtype= np.float32)
		obj = y_train*y_ - np.exp(y_) - np.log(fact)
		return -np.sum(obj)

	return objective_fn

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


df = pd.read_csv('./horsekicks.csv')

x_train = np.asarray([ [math.pow(year/20.0,p) for p in [0,0.5,1.0, 1.5]] for year in range(0,13)], dtype=np.float32)
x_test = np.asarray([[math.pow(year/20.0,p) for p in [0,0.5,1.0, 1.5]] for year in range(13,20)], dtype=np.float32)
# x_train = x_train/20.0
# x_test = x_test/20.0

for corps_idx in range(14):
	print('Corp : ',df.iloc[corps_idx,0])
	y_train = np.asarray(df.iloc[corps_idx,1:14], dtype=np.uint32)
	y_test = np.asarray(df.iloc[corps_idx,14:-1], dtype=np.uint32)

	theta = np.random.randn(4)

	obj_fn = get_objective(x_train, y_train)
	res = minimize(obj_fn, theta, method='l-bfgs-b', options={'maxiter': 1000})

	theta_ = res.x
	print('Solution : ', res)
	pred_test = np.exp(np.dot(x_test,theta_))
	print('Predicted : ', pred_test)
	print('Target : ', y_test)

	print('RMSE : ', rmse(pred_test, y_test))
	print()


