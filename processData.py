'''
Basically in the HAM10000 dataset their is a very large class imbalance:
nv, 6705
mel, 1113
bkl, 1099
bcc, 514
akiec, 327
vasc, 142
df, 115

and so basically the 3 largest classes takes up 89% of all the images
so instead the model only trains on the 3 largest classes 
and also makes each class the same size so all of them 1099 size * 3

so the dataset goes from 10000 to 3297

but the images are flipped using data augmentation, so the dataset quadruples to 13188 

also the images are split into a 80/20 split where 80% is training and 20% is testing
'''
from quacknet import Augementation
import pandas as pd
import numpy as np
import os
import random

augmentation = Augementation()
imagePaths = augmentation.getImagePaths('rawData/data/')
random.shuffle(imagePaths)

metadata = pd.read_csv('rawData/HAM10000_metadata.csv')
labelsDict = dict(zip(metadata['image_id'], metadata['dx']))

labelIdDict = {
    "nv": 0,      #largest
    "mel": 1,     #second largest
    "bkl": 2,     #third largest
    "akiec": 3,
    "bcc": 4,
    "df": 5,
    "vasc": 6,
}

topClasses = ["nv", "mel", "bkl"]

labels = []
images = []
for i in imagePaths:
    imageId = os.path.basename(i)[:-4] # removes the .jpg extsension
    label = labelsDict[imageId]
    if(label in topClasses):
        labels.append(labelIdDict[label])
        images.append(i)
imagePaths = images
labels = augmentation.hotEncodeLabels(labels, 3) #only 3 classes

processedImage = augmentation.preprocessImages(imagePaths, (64, 64))
images = processedImage.transpose(0, 3, 1, 2)
labelsAugmented = labels 

count = np.zeros(len(labelsAugmented[0]))
for i in range(len(labelsAugmented)):
    count = count + np.array(labelsAugmented[i])

sorted = np.argsort(count)
largestClass = sorted[-1] #largest class index
secondClass = sorted[-2]  #second largest class index
thirdClass = sorted[-3]   #third largest class index

thirdLargestClass = int(count[thirdClass]) #gets the size of the 3rd largest class

allClasses = [[], [], []]
for i in range(len(images)):
    allClasses[np.argmax(labelsAugmented[i])].append((images[i], labelsAugmented[i]))

allImages = []
allLabels = []
for i in range(len(allClasses)):
    if(i == largestClass or i == secondClass or i == thirdClass):
        samples = allClasses[i][:thirdLargestClass]
        allImages.extend([sample[0] for sample in samples])
        allLabels.extend([sample[1] for sample in samples])

newImages, newLabels = augmentation.dataAugmentation(allImages, allLabels) #augmentates the images by flipping them which makes the dataset quadruple in size

indexToSplitAt = len(allImages) * 8 // 10    # splits the images into 80/20 split

np.save('processedData/training/ham10000_images.npy', np.array(allImages[0: indexToSplitAt - 1], dtype=np.float32))
np.save('processedData/training/ham10000_labels.npy', np.array(allLabels[0: indexToSplitAt - 1]))

np.save('processedData/test/ham10000_images.npy', np.array(allImages[indexToSplitAt: len(allImages) - 1], dtype=np.float32))
np.save('processedData/test/ham10000_labels.npy', np.array(allLabels[indexToSplitAt: len(allImages) - 1]))
