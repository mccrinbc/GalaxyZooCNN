import shutil
import torch 

PATH = "/Users/brianmccrindle/Documents/736/galaxy-zoo-the-galaxy-challenge"
IDs = torch.load(PATH + '/IDs.pt')
classLabels = torch.load(PATH + '/classLabels.pt')

curImgPath = PATH + '/preprocessed/'

#count the number of images in each class
C1 = 0
C2 = 0
C3 = 0

for ii in range(len(IDs)):
	name = str(IDs[ii].item()) + '.jpg'
	if classLabels[ii][0] == 1:
		shutil.move(curImgPath + name, curImgPath + 'C1/' + name)
		C1 += 1
	elif classLabels[ii][1] == 1:
		shutil.move(curImgPath + name, curImgPath + 'C2/' + name)
		C2 += 1
	else:
		shutil.move(curImgPath + name, curImgPath + 'C3/' + name)
		C3 += 1
	