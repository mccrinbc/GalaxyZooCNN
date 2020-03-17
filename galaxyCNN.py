import os 
from tqdm import tqdm #loading bar 
import numpy as np 
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import torch.optim as optim

dir_path = '/Users/brianmccrindle/Documents/736/galaxy-zoo-the-galaxy-challenge'
os.chdir(dir_path) #change to this working directory 
classes = ('Smooth','Disk','Artifact')

#imgIDs = torch.load(dir_path +'/IDs.pt')

#class to call the data from local machine
class buildImgStack():
	classLocs = os.getcwd() + '/preprocessed/'
	imgIDs = torch.load(os.getcwd() + '/IDs.pt')
	classLabels = torch.load(os.getcwd() + '/classLabels.pt')

	transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), 
                                    transforms.ToTensor()])

	def load_dataset(self, type):
	    if type == 'train':
	    	data_path = self.classLocs + 'train/'
	    	shuffle_status = True
	    elif type == 'test':
	    	data_path = self.classLocs + 'test/'
	    	shuffle_status = False

	    dataset = torchvision.datasets.ImageFolder(
	        root = data_path,
	        transform = self.transform
	    )
	    dataLoader = DataLoader(
	        dataset,
	        batch_size = 64,
	        num_workers = 0, #parallel processing
	        shuffle = shuffle_status
	    )
	    #classes are sorted in alphabetical order from folders when using dataLoader

	    #classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
		#classes.sort()
	    return dataLoader



# Building a simple multi-layer CNN for image classification 
class convNet(nn.Module):
	def __init__(self):
		super().__init__()
					#nn.Conv2d(input_channels, output_channels, kernal_size)
		self.conv1 = nn.Conv2d(1,  32,  3, stride = 1, padding = 0, bias = True) 
		self.conv2 = nn.Conv2d(32, 64,  3, stride = 1, padding = 0, bias = True) 
		self.conv3 = nn.Conv2d(64, 128, 3, stride = 1, padding = 0, bias = True) 

		#this is to get the size of the input to FCL
		x = torch.randn([191,191]).view(-1,1,191,191)
		self.input = None
		self.convolve(x)

		#the size of the input (output from conv3 should be 185)
		self.fc1   = nn.Linear(self.input,128) 
		self.fc2   = nn.Linear(128,3) #three output classes (Smooth, Disk, Artifact)

	def convolve(self,x):
		x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)) #pooling layers 2x2
		x = F.max_pool2d(F.relu(self.conv2(x)),(2,2))
		x = F.max_pool2d(F.relu(self.conv3(x)),(2,2))

		if self.input is None:
			#input shape to the fully connected layer
			self.input = x[0].shape[0]*x[0].shape[1]*x[0].shape[2]
			#print(self.input)
		return x

	def forward(self,x):
		x = self.convolve(x)
		x = x.view(-1, self.input) #flatten the image for the fully-connected layers

		x = F.relu(self.fc1(x))
		x = self.fc2(x)

		#return F.softmax(x, dim = 1) #the x in this case could be a batch of x's
		return x #only need x if CrossEntropyLoss


model = convNet()
images = buildImgStack()
dataLoaderTrain = images.load_dataset(type = 'train')
dataLoaderTest =  images.load_dataset(type = 'test' )

#activate the GPU if avaliable 
if torch.cuda.is_available():
	device = torch.device("cuda:0")
	print("Running on GPU")
else:
	device = torch.device("cpu")
	print("Running on CPU")

model = convNet().to(device)
print(model)

def trainModel(model):
	#we need to place the optimizer and criterion in the function if we
	#want to run the model on the GPU. 
	EPOCHS = 5 #for now 
	batchLoss = []
	batchIndex = []

	optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum = 0.90)
	criterion = nn.CrossEntropyLoss() #this might not work?
	for epoch in range(EPOCHS):
		for batch_idx, (data, target) in enumerate(dataLoaderTrain):

			#for GPU use
			data.to(device)
			target.to(device)
			print(batch_idx)

			#This is a bit of a hack but whatever
			#need to rearrange the labels to match what dataloader expects
			labels = []
			for j in range(0,len(target)):
				labels.append(np.eye(3)[target[j].item()])

			labels = torch.Tensor(labels) #need to convert the labels into a tensor

			model.zero_grad()
			output = model(data) #data.size() = [64,1,191,191]
			#return output, data, labels
			loss = criterion(output,target)
			loss.backward()
			optimizer.step()

			batchLoss.append(loss)
			batchIndex.append(batch_idx)

			print(f"Batch Loss: {loss}")
	print(f"Epoch: {epoch}. Loss: {loss}")
	return batchLoss, batchIndex

TRAIN = False
if TRAIN == True:
	batchLoss, batchIndex = trainModel(model)
	#Depending on the use case, we can either save the entire model (arch + weights)
	#or just the weights. The line below is for saving ONLY the weights.
	#This lets us call the untrained model when we want. 
	torch.save(model.state_dict(), dir_path + '/trained_model_weights.pt') 
	print('saved model weights')

TEST = True
if TEST == True:
	testModelNet = convNet().to(device)
	testModelNet.load_state_dict(torch.load(dir_path + '/trained_model_weights.pt'))
	print('Loaded Weights into testModelNet')

	def testModel(testModelNet):
		#record the number of correct classifications
		class_correct = list(0. for i in range(3))
		class_total = list(0. for i in range(3))
		with torch.no_grad():
			for batch_idx, (testData, target) in enumerate(dataLoaderTest):
				#for GPU use
				testData.to(device)
				target.to(device)

				output = testModelNet(testData)
				_, predicted = torch.max(output.data,1)
				print(predicted)
				print(target)
				results = (predicted == target).squeeze() #might not need sqeeze

				print(f'batch_idx: {batch_idx}')
				for ii in range(0,len(target)): #should always be 64
					label = target[ii]
					#below is a smart way of adding if TRUE or not if FALSE
					class_correct[label] += results[ii].item()
					class_total[label] += 1
				print(f'Class Correct Array: {class_correct}')
				print(f'Class Total Array: {class_total}')

		for ii in range(3):
			print(classes[ii], 100 * class_correct[ii] / class_total[ii])

		return class_correct, class_total

	class_correct, class_total =  testModel(model)







