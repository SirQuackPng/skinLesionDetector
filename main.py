from quacknet import Conv2DLayer, PoolingLayer, DenseLayer, ActivationLayer, GlobalAveragePooling
from quacknet import Network, CNNModel
import numpy as np
import time

learningRate = 0.0000625
batchSize = 32

depths = [3, 64, 128]
numKernals = [64, 128, 256] 

# creates a neural network class, with no hidden layers
NN = Network(lossFunc = "cross", learningRate = learningRate)
NN.addLayer(256)
NN.addLayer(3, "softmax")

# creates a convulational neural network class
cnn = CNNModel(NN)
for i in range(len(depths)): #creates 2 convulutional blocks
    cnn.addLayer(Conv2DLayer(
        kernalSize = 3,
        depth = depths[i],
        numKernals = numKernals[i],
        stride = 1, 
        padding = "no" 
    ))
    cnn.addLayer(ActivationLayer())
    cnn.addLayer(PoolingLayer(
        gridSize = 2,
        stride = 2,
        mode = "max", #using max pooling
    ))

cnn.addLayer(GlobalAveragePooling())
cnn.addLayer(DenseLayer(NN))

#loads weights and biases from a file
cnn.loadModel(NN)

#loads training images and labels
images = np.load("processedData/training/ham10000_images.npy", mmap_mode='r').astype(np.float32)
labels = np.load("processedData/training/ham10000_labels.npy", mmap_mode='r').astype(np.float32)

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

"""
accauracy: 
48.043676069153776, 52.4416135881104, 48.801941158629056, 48.34698210494388, 50.621777373369724, 51.8956627236882, 53.472854109796785, 53.07855626326964, 53.92781316348195, 54.382772217167116, 55.868971792538666, 55.44434334243251, 55.99029420685472, 61.20715802244465, 62.26872914771005, 64.51319381255686, 63.512283894449496, 66.87898089171973, 67.94055201698514, 68.30451925993327, 68.9111313315135, 71.51956323930845, 73.09675462541705, 72.975432211101, 74.21898695784046, 74.64361540794661, 74.58295420078859, 75.58386411889597, 75.55353351531696, 75.09857446163178

loss: 
11.740347163605941, 8.829523506523469, 5.085878912073869, 3.2873291547680883, 2.322169851860588, 1.8738625077713869, 1.5269737029126325, 1.4394840805810964, 1.434182773798531, 1.2932932289878216, 1.2534780258073073, 1.2688587728527734, 1.241306545132356, 0.937132778541741, 0.8853267268705293, 0.8603156508839079, 0.8613275027028133, 0.7544607297593156, 0.738125630946445, 0.7315617589702545, 0.7186301969197343, 0.6576075921624474, 0.6403682137652609, 0.6280996254118703, 0.6006174283373661, 0.5917234915453385, 0.5874717859329187, 0.5815756053687812, 0.5774836305951975, 0.5705471087801189

LR reduced from 0.001 to 0.0005 on epoch 14 (seen from increase of acurracy 55.99029420685472 to 61.20715802244465)
LR reduced from 0.0005 to 0.00025 on epoch 18
LR reduced from 0.00025 to 0.000125 on epoch 22
LR reduced from 0.000125 to 0.0000625 on epoch 25 (72.975432211101 to 74.21898695784046)


test accauracy:
72.3823975720789

test loss:
0.7442537257896777
"""