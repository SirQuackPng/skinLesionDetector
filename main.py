import quacknet
import quacknet.convulationalManager
from quacknet.convulationalManager import ConvLayer, PoolingLayer, DenseLayer, ActivationLayer
from quacknet.main import Network
import numpy as np
import time

learningRate = 0.00025
batchSize = 64

depths = [3, 64]
numKernals = [64, 128] 

# creates a neural network class, with no hidden layers
NN = Network(lossFunc = "cross", learningRate = learningRate)
NN.addLayer(128, "relu")
NN.addLayer(3, "softmax")

# creates a convulational neural network class
cnn = quacknet.convulationalManager.CNNModel(NN)
for i in range(2): #creates 2 convulutional blocks
    cnn.addLayer(ConvLayer(
        kernalSize = 3,
        depth = depths[i],
        numKernals = numKernals[i],
        stride = 1, 
        padding = "n"  #no padding
    ))
    cnn.addLayer(ActivationLayer())
    cnn.addLayer(PoolingLayer(
        gridSize = 2,
        stride = 2,
        mode = "max", #using max pooling
    ))

cnn.addLayer(PoolingLayer(
    None, #doesnt matter since GAP doesnt have grid size
    None, #doesnt matter since GAP doesnt have stride length
    mode = "gap" # Global Average Pooling or (GAP)
))
cnn.addLayer(DenseLayer(
    NN
))

#loads weights and biases from a file
cnn.loadModel(NN)

#loads training images and labels
images = np.load("processedData/training/ham10000_images.npy", mmap_mode='r')
labels = np.load("processedData/training/ham10000_labels.npy", mmap_mode='r')

# randomly shuffles images and labels, but makes it where the image corresponds to the right label
indices = np.random.permutation(len(images))
images = images[indices]
labels = labels[indices]

print("started") 
start = time.time()
accauracy, loss = cnn.train(
    inputData=images,
    labels=labels,
    useBatches=True,
    batchSize=batchSize,
    alpha=learningRate,
)

print(f"took: {(time.time() - start)} seconds or {(time.time() - start) / 60} minutes")
print(f"accauracy: {accauracy}")
print(f"loss: {loss}")

#saves weights and biases
cnn.saveModel(NN.weights, NN.biases, cnn.weights, cnn.biases)
