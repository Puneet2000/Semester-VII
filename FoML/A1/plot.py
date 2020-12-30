import pandas as pd 
import numpy as np 
import math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
# np.random.seed(1)

def likelihood(y_train):

	def fn(lam): # Poisson Likelihood
		n = len(y_train)

		y_fact = [math.factorial(y) for y in y_train]
		return np.exp(-n*lam)*np.power(lam, np.sum(y_train))/np.prod(y_fact)

	return fn

def posterior(y_train, alpha=1.0, beta=0.5): # Posterior

	def fn(lam):
		n = len(y_train)

		y_fact = [math.factorial(y) for y in y_train]
		return np.power(beta,alpha)*np.exp(-(n+beta)*lam)*np.power(lam, np.sum(y_train) + alpha -1.0 )/np.prod(y_fact)*math.gamma(alpha)

	return fn

def prior(alpha=1.0, beta=0.5): # Gamma Prior

	def fn(lam):
		return np.power(beta,alpha)*np.power(lam,alpha-1)*np.exp(-beta*lam)/math.gamma(alpha)

	return fn

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


df = pd.read_csv('./horsekicks.csv')


for corps_idx in [2,4,6]:
	print('Corp : ',df.iloc[corps_idx,0])
	y_train = np.asarray(df.iloc[corps_idx,1:14], dtype=np.uint32)
	y_test = np.asarray(df.iloc[corps_idx,14:-1], dtype=np.uint32)


	likelihood_fn = likelihood(y_train)
	posterior_fn = posterior(y_train, alpha=3.0, beta=2.0)
	prior_fn = prior(alpha=3.0, beta=2.0)

	ws = np.linspace(0,10,10000) # sample values from 0 to 10
	likelihood_y  = [ likelihood_fn(w) for w in ws]
	posterior_y  =[ posterior_fn(w) for w in ws]
	prior_y  = [ prior_fn(w) for w in ws]

	likelihood_y = likelihood_y/np.sum(likelihood_y) # normalize values so that they are visible to same extent in graph
	posterior_y = posterior_y/np.sum(posterior_y)
	prior_y = prior_y/np.sum(prior_y)


	fig = plt.figure()
	plt.plot(ws,likelihood_y, label='Likelihood')  # plot
	plt.plot(ws,posterior_y, label='Posterior')
	plt.plot(ws,prior_y, label='Prior')
	plt.legend()

	fig.savefig('./corp{}.png'.format(corps_idx)) # save figures



