from quacknet.dataAugmentation import Augementation
import pandas as pd
import numpy as np
import os

augmentation = Augementation()
imagePaths = augmentation.getImagePaths('rawData/data/')

metadata = pd.read_csv('rawData/HAM10000_metadata.csv')
labelsDict = dict(zip(metadata['image_id'], metadata['dx']))

labelIdDict = {
    "nv": 0,
    "mel": 1,
    "bcc": 2,
    "akiec": 3,
    "bkl": 4,
    "df": 5,
    "vasc": 6,
}

labels = []
for i in imagePaths:
    imageId = os.path.basename(i)[:-4] # removes the .jpg extsension
    labels.append(labelIdDict[labelsDict[imageId]])
labels = augmentation.hotEncodeLabels(labels, 7)

processedImage = augmentation.preprocessImages(imagePaths, (128, 128))
images = augmentation.dataAugmentation(processedImage)
images = images.transpose(0, 3, 1, 2)

labelsAugmented = np.repeat(labels, 4, axis=0)

np.save('processedData/ham10000_images.npy', np.array(images, dtype=np.float32))
np.save('processedData/ham10000_labels.npy', np.array(labelsAugmented))


