import quacknet
import quacknet.convulationalManager
from quacknet.convulationalManager import ConvLayer, PoolingLayer, DenseLayer, ActivationLayer
from quacknet.main import Network
import numpy as np
import time

# creates a neural network class
NN = Network(lossFunc = "cross", learningRate = 0.00025)
NN.addLayer(128, "relu")
NN.addLayer(3, "softmax")

#depths = [3, 32, 64]
#numKernals = [32, 64, 128] 

depths = [3, 64]
numKernals = [64, 128] 

# creates a convulational neural network class
cnn = quacknet.convulationalManager.CNNModel(NN)
for i in range(2): 
    cnn.addLayer(ConvLayer(
        kernalSize = 3,
        depth = depths[i],
        numKernals = numKernals[i],
        stride = 1, 
        padding = "n"
    ))
    cnn.addLayer(ActivationLayer())
    cnn.addLayer(PoolingLayer(
        gridSize = 2,
        stride = 2,
        mode = "max",
    ))

cnn.addLayer(PoolingLayer(
    None, #doesnt matter since GAP doesnt have grid size
    None, #doesnt matter since GAP doesnt have stride length
    mode = "gap" # Global Average Pooling or (GAP)
))
cnn.addLayer(DenseLayer(
    NN
))


cnn.createWeightsBiases()
NN.createWeightsAndBiases()
cnn.saveModel(NN.weights, NN.biases, cnn.weights, cnn.biases)

cnn.loadModel(NN)

images = np.load("processedData/training/ham10000_images.npy", mmap_mode='r')
labels = np.load("processedData/training/ham10000_labels.npy", mmap_mode='r')

# randomly shuffles images and labels
indices = np.random.permutation(len(images))
images = images[indices]
labels = labels[indices]

batchSize = 64
print("started") 
start = time.time()
accauracy, loss = cnn.train(
    inputData=images,
    labels=labels,
    useBatches=True,
    batchSize=batchSize,
    alpha=0.00025,
)
print(f"took: {(time.time() - start)} seconds or {(time.time() - start) / 60} minutes")
print(f"accauracy: {accauracy}")
print(f"loss: {loss}")


cnn.saveModel(NN.weights, NN.biases, cnn.weights, cnn.biases)

'''
accauracy:


loss:

'''