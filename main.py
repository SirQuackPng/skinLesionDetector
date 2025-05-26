import quacknet
import quacknet.convulationalManager
from quacknet.convulationalManager import ConvLayer, PoolingLayer, DenseLayer, ActivationLayer
from quacknet.main import Network
import numpy as np
import time

# creates a neural network class
NN = Network(lossFunc = "cross", learningRate = 0.001)
NN.addLayer(128, "relu")
NN.addLayer(7, "softmax")

# creates a convulational neural network class
cnn = quacknet.convulationalManager.CNNModel(NN)
# repeates Convolutional Block 4 times
for i in range(4): 
    cnn.addLayer(ConvLayer(
        kernalSize = 3,
        depth = 3,
        numKernals = 32,
        stride = 1, 
        padding = "1"
    ))
    cnn.addLayer(ActivationLayer())
    cnn.addLayer(PoolingLayer(
        gridSize = 2,
        stride = 1,
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

images = np.load("processedData\ham10000_images.npy", mmap_mode='r')
labels = np.load("processedData\ham10000_labels.npy", mmap_mode='r')

print("started")
start = time.time()
accauracy, loss = cnn.train(
    inputData=images[:32],
    labels=labels[:32],
    useBatches=True,
    batchSize=32
)

print(f"took: {time.time() - start} seconds")
print(accauracy, loss)