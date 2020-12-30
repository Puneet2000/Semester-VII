"""
To run: python3 main.py

"""
import torch
import pandas as pd 
import numpy as np 
import torch.utils.data as dset
import torch.nn as nn
import torch.optim as optim
import copy
import matplotlib.pyplot as plt 

np.random.seed(0)
torch.random.manual_seed(0) 

# Load data
df = pd.read_csv('./data.csv')

# define all Costs
fnc = df.iloc[:,10]
fpc, tpc, tnc = 150.0, 150.0, 0.0 

# convert data to Pytorch Tensors
df = pd.concat([df.iloc[:,:10], df.iloc[:,11:]], axis=1)
data, targets = df.iloc[:,1:], df.iloc[:,0]
data, targets, fnc = torch.FloatTensor(data.to_numpy()), torch.LongTensor(targets.to_numpy()), torch.FloatTensor(fnc.to_numpy())

# Split into train, validation and test data
dataset = dset.TensorDataset(data, targets, fnc)
trainset, valset, testset = dset.random_split(dataset, [100000, 15000, 32636])
trainloader = dset.DataLoader(trainset, batch_size=1024, shuffle=True)
valloader = dset.DataLoader(valset, batch_size=1024, shuffle=False)
testloader = dset.DataLoader(testset, batch_size=1024, shuffle=False)

# Define Logistic Regressor
net = nn.Sequential( nn.Linear(11,1),
					 nn.Sigmoid())


# optimizer to optimize weights
optimizer = optim.Adam(net.parameters())

num_epochs = 20
best_acc = 0
loss_train, loss_val = [], []
acc_train, acc_val = [], []
for epoch in range(num_epochs):
	net.train()
	losses = []
	correct, total = 0, 0
	# Start training
	for i, (x, y, cfn ) in enumerate(trainloader):
		optimizer.zero_grad()

		hx = net(x).squeeze(1) # calculate current model prediction

		pred = (hx>0.5).long()
		correct += (pred==y).sum().item()
		total += hx.size(0)

		loss = y*(hx*tpc + (1-hx)*cfn) + (1-y)*(hx*fpc + (1-hx)*tnc) # calculate cost sensitive Loss
		loss = loss.mean()

		loss.backward() # calculate gradients
		losses.append(loss.data.item())
		optimizer.step()

	acc_train.append(100*correct/total)
	loss_train.append(np.mean(losses))

	net.eval()
	losses = []
	correct, total = 0, 0
	for i, (x, y, cfn ) in enumerate(valloader): # evaluate on validation test
		hx = net(x).squeeze(1)

		pred = (hx>0.5).long()
		correct += (pred==y).sum().item()
		total += hx.size(0)

		loss = y*(hx*tpc + (1-hx)*cfn) + (1-y)*(hx*fpc + (1-hx)*tnc)
		loss = loss.mean()
		losses.append(loss.data.item())

	acc_val.append(100*correct/total)
	loss_val.append(np.mean(losses))

	if acc_val[-1] > best_acc: # if val acc is better, save checkpoint
		torch.save(net.state_dict(), './best.pth')
		best_acc = acc_val[-1]


	print('Epoch {}, Train Loss {:3f}, Val Loss {:3f}, Train accuracy {:.3f}, Val accuracy {:.3f}'.format(epoch, loss_train[-1], loss_val[-1], acc_train[-1], acc_val[-1]))


net.eval()
net.load_state_dict(torch.load('./best.pth'))
correct, total = 0, 0
tp, fp, fn = 0, 0, 0
for i, (x, y, cfn ) in enumerate(testloader): # evaluate on test data
	hx = net(x).squeeze(1)

	# for accuracy, find correct predictions
	pred = (hx>0.5).long()
	correct += (pred==y).sum().item()
	total += hx.size(0)

	# find true positives, false positives, false negatives
	tp += ((pred == 1)*(y==1)).sum().item() 
	fp += ((pred==1)*(y==0)).sum().item()
	fn += ((pred==0)*(y==1)).sum().item()

# Find accuracy, precision, recall, F1 Score
prec, rec = 1.0*tp/(tp+fp), 1.0*tp/(tp+fn)
f1 = 2*prec*rec/(prec + rec)

print('Test accuracy {:.3f}, Precision {:.3f}, Recall {:.3f}, F1 {:.3f}'.format(100.*correct/total, prec, rec, f1))

# plot train/Val accuracy/loss graphs
plt.figure()
plt.plot(range(num_epochs), loss_train, label = 'Train', marker='o')
plt.plot(range(num_epochs), loss_val, label = 'Val', marker='s')
plt.title('Train vs Val Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend()
plt.grid()
plt.savefig('./loss.png')

plt.figure()
plt.plot(range(num_epochs), acc_train, label = 'Train', marker='o')
plt.plot(range(num_epochs), acc_val, label = 'Val', marker='s')
plt.title('Train vs Val Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.grid()
plt.savefig('./accuracy.png')
