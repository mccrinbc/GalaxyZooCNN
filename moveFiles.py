import shutil
import torch 

PATH = "/Users/brianmccrindle/Documents/736/galaxy-zoo-the-galaxy-challenge"
IDs = torch.load(PATH + '/IDs.pt')
classLabels = torch.load(PATH + '/classLabels.pt') #these are randomally sorted already

trainSize = round(len(classLabels)*0.70) #70% of data for train

IDTrain = IDs[0:trainSize] 
IDTest  = IDs[trainSize+1 : len(classLabels)]
classLabelsTrain = classLabels[0:trainSize] 
classLabelsTest  = classLabels[trainSize+1 : len(classLabels)]

curImgPath = PATH + '/preprocessed/'

#for training organizatiopn
for ii in range(len(IDTrain)):
	name = str(IDTrain[ii].item()) + '.jpg'
	if classLabelsTrain[ii][0] == 1:
		shutil.move(curImgPath + 'C1/' + name, curImgPath + 'train/C1/' + name)
	elif classLabelsTrain[ii][1] == 1:
		shutil.move(curImgPath + 'C2/' + name, curImgPath + 'train/C2/' + name)
	else:
		shutil.move(curImgPath + 'C3/' + name, curImgPath + 'train/C3/' + name)

#for testing organizatiopn
for ii in range(len(IDTest)):
	name = str(IDTest[ii].item()) + '.jpg'
	if classLabelsTest[ii][0] == 1:
		shutil.move(curImgPath + 'C1/' + name, curImgPath + 'test/C1/' + name)
	elif classLabelsTest[ii][1] == 1:
		shutil.move(curImgPath + 'C2/' + name, curImgPath + 'test/C2/' + name)
	else:
		shutil.move(curImgPath + 'C3/' + name, curImgPath + 'test/C3/' + name)
	