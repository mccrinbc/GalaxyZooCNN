import numpy as np
import os
import pandas as pd 
import torch

#function made to change the labels of the training/testing set into one-hot encoded labels.
#The given testing set of images for the GalaxyZoo dataset doesn't have lables. Therefore,
#we're going to split the training set into 70/30. 
PATH = "/Users/brianmccrindle/Documents/736/galaxy-zoo-the-galaxy-challenge"

os.chdir(PATH)
data = pd.read_csv("training_solutions_rev1.csv")

data.head() #display purposes

IDs = data.iloc[:]['GalaxyID']
C1 = data.iloc[:]['Class1.1']
C2 = data.iloc[:]['Class2.2']
C3 = data.iloc[:]['Class1.3']
 
 #this isn't clean but we can change this in the future!
labels = np.zeros((len(C1),3))
labels = torch.from_numpy(labels) #making this into a tensor, ease of computation 

for ii in range(len(C1)):
	values = torch.Tensor([C1[ii], C2[ii], C3[ii]])
	index = torch.argmax(values)
	if index == 0:
		values = torch.Tensor([1,0,0])
	elif index == 1:
		values = torch.Tensor([0,1,0])
	else:
		values = torch.Tensor([0,0,1])

	labels[ii] = values

torch.save(labels, 'classLabels.pt')
torch.save(torch.tensor(IDs), 'IDs.pt')
#torch.load('file.pt')