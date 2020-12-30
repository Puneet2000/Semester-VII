"""
To run: python3 main.py

"""
import torch
import pandas as pd 
import numpy as np 
import torch.utils.data as dset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans
from collections import Counter

color_dict = {0: 'red', 1: 'green', 2:'orange', 3: 'blue', 4: 'brown'}
K = 2
np.random.seed(0)
torch.random.manual_seed(0) 

# Read data csv file
df = pd.read_csv('./data.csv')
data = torch.FloatTensor(df.to_numpy())

# Model architecture of VAE
class VAE(nn.Module):
	def __init__(self, out_dim=9):
		super(VAE, self).__init__()
		self.encoder  = nn.Sequential( nn.Linear(out_dim, out_dim//2),
										nn.ReLU(),
										nn.Linear(out_dim//2, 2*out_dim//4),
				)

		self.decoder = nn.Sequential( nn.Linear(out_dim//4, out_dim//2),
										nn.ReLU(),
										nn.Linear(out_dim//2, out_dim))

	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		esp = torch.randn(*mu.size())
		z = mu + std * esp
		return z
	
	def forward(self, x):
		h = self.encoder(x)
		mu, logvar = torch.chunk(h, 2, dim=1) # find mean and logvar
		z = self.reparameterize(mu, logvar) # reparametrize to get Noemal(mean, var)
		xr = self.decoder(z)
		return xr, mu, logvar

net = VAE(9)
optimizer = optim.Adam(net.parameters()) # Adam Optimizer

num_epochs = 1000
for epoch in range(num_epochs):
	net.train()
	xr, mu, logvar = net(data) # Find reconstructed image, mean and logvar
	loss = F.mse_loss(xr,data) -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) # Optimize MSE and KL Divergence
	optimizer.zero_grad()
	loss.backward() # caluclate gradients
	optimizer.step() # optimize

	if epoch%100 == 0:
		print('Epoch {}, Train Loss {:3f}'.format(epoch, loss.data.item()))


net.eval()
h = net.encoder(data) # calculate VAE hidden dimesion
data_mu, data_logvar = torch.chunk(h, 2, dim=1)
data_mu = data_mu.detach().numpy()
# print(data_mu)
kmeans = KMeans(n_clusters=K, max_iter=1000, random_state=0).fit(data_mu) # cluster on VAE representation
plt.figure()
classes = kmeans.predict(data_mu)
cnt = Counter(classes)
print(cnt)
# print((classes==0).sum())
for i in range(data_mu.shape[0]):
	plt.scatter(data_mu[i,0], data_mu[i,1], color=color_dict[classes[i].item()])
plt.title('Clustering on VAE encoder space ( K = {} )'.format(K))
plt.legend()
plt.savefig('./cluster{}.png'.format(K))
