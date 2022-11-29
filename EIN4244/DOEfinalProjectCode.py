#####Modified DOE Final Project Code
#####Edited by: Christopher Kilroy
#####Last edited: 28 November 2022
#####This code is used to calculate optimal hyperparameters for training a convolutional neural network
#####Modifications include:
##### 1. defining functions to initialize hyperparameter values, 
##### 2. automatic calculation of parameter extrema (min and max),
##### 3. batch processing to calculate accuracy values
#####Collecting accuracy values is only semi-automated at this stage; calculated values will need to be copied from the printout


###Import libraries
import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import optimizers
from tensorflow.keras.datasets import cifar10

###Load data
(X_train, y_train), (X_val, y_val) = cifar10.load_data()
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size= 0.25)
X_train=X_train[:20000,:,:,:]
y_train=y_train[:20000,:]
X_val=X_val[:1000,:,:,:]
y_val=y_val[:1000,:]

labels=dict()
labels["0"]="airplane"
labels["1"]="automobile"
labels["2"]="bird"
labels["3"]="cat"
labels["4"]="deer"
labels["5"]="dog"
labels["6"]="frog"
labels["7"]="horse"
labels["8"]="ship"
labels["9"]="truck"

index=np.random.choice(X_train.shape[0], 4, False)
# Lets see the dataset!
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[index[0]], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[index[1]], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[index[2]], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[index[3]], cmap=plt.get_cmap('gray'))

# show the plot
plt.show()

# scale pixels
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm


X_train, X_val=prep_pixels(X_train, X_val)
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
activation = 'relu'

###Range of the hyperparameters
#Kernel (filter) size Conv2D: 1, 3, 5, 7, 11 (integer)
#Number of convolutional layers = 1 - 4 (integer)
#Number of filter conv2D = 16, 32, 64, 128, 264 (integer)
#Number of neurons = 50 - 300 (integer)
#Learning rate= 0.0001 - 1 (continuous)
#Number of epochs: 1 - 40
#Batch size = 1-100 (integer)
#momentum = 0.5 - 0.99 (continuous)

###Initialize hyperparameter values
def resetParameters():
  #initial parameters
  kernel_size_Conv2D = 11 #a
  number_of_convolutional_layers = 1  #b
  number_of_filter_conv2D = 64  #c
  number_of_neurons = 200 #d
  learning_rate = 0.0001  #e
  number_of_epochs = 10 #f
  batch_size = 10 #g
  momentum = 0.8  #h

  return kernel_size_Conv2D, number_of_convolutional_layers, number_of_filter_conv2D, number_of_neurons, learning_rate, number_of_epochs, batch_size, momentum


###Train the convolutional neural network
def GetValAccuracy(kernel_size_Conv2D,number_of_convolutional_layers,number_of_filter_conv2D,number_of_neurons,learning_rate,number_of_epochs,batch_size,momentum):

  model = Sequential()

  model.add(Conv2D(number_of_filter_conv2D, (kernel_size_Conv2D, kernel_size_Conv2D), activation = activation, kernel_initializer='he_uniform', padding='same', input_shape=(32, 32, 3)))
  model.add(MaxPooling2D((3, 3)))

  if number_of_convolutional_layers >= 2:
    model.add(Conv2D(number_of_filter_conv2D, (kernel_size_Conv2D, kernel_size_Conv2D), activation = activation, kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((3, 3)))

  if number_of_convolutional_layers >= 3:
    model.add(Conv2D(number_of_filter_conv2D, (kernel_size_Conv2D, kernel_size_Conv2D), activation = activation, kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((3, 3)))

  if number_of_convolutional_layers >= 4:
    model.add(Conv2D(number_of_filter_conv2D, (kernel_size_Conv2D, kernel_size_Conv2D), activation = activation, kernel_initializer='he_uniform', padding='same'))

  model.add(Flatten())
  model.add(Dense(number_of_neurons, activation = activation, kernel_initializer='he_uniform'))
  model.add(Dense(10, activation='softmax'))

  opt = SGD(lr = learning_rate, momentum = momentum)
  model.compile(loss = 'categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
  #print(model.summary())
  model.fit(X_train, y_train, epochs = number_of_epochs, batch_size = batch_size, validation_data = (X_val, y_val))
    

  val_accuracy = model.evaluate(X_val, y_val)[1]

  return val_accuracy


##get baseline accuracy
kernel_size_Conv2D, number_of_convolutional_layers, number_of_filter_conv2D, number_of_neurons, learning_rate, number_of_epochs, batch_size, momentum = resetParameters()

baselineAccuracy = GetValAccuracy(kernel_size_Conv2D,number_of_convolutional_layers,number_of_filter_conv2D,number_of_neurons,learning_rate,number_of_epochs,batch_size,momentum)
print(f"Baseline Accuracy: {baselineAccuracy}")

##get kernel size accuracies
kernelsizeRange = [1,3,5,6,11]
kernelsizeAccuracy = []

kernel_size_Conv2D, number_of_convolutional_layers, number_of_filter_conv2D, number_of_neurons, learning_rate, number_of_epochs, batch_size, momentum = resetParameters()

for kernel_size_Conv2D in kernelsizeRange:
  val_accuracy = GetValAccuracy(kernel_size_Conv2D,number_of_convolutional_layers,number_of_filter_conv2D,number_of_neurons,learning_rate,number_of_epochs,batch_size,momentum)
  kernelsizeAccuracy.append(val_accuracy)

i = 0
for accuracy in kernelsizeAccuracy:
  if accuracy == min(kernelsizeAccuracy):
    kernelsizeLow = kernelsizeRange[i]
    kernelsizeAccuracyLow = accuracy
  if accuracy == max(kernelsizeAccuracy):
    kernelsizeHigh = kernelsizeRange[i]
    kernelsizeAccuracyHigh = accuracy 
  i += 1

print(f"Kernel size low: {kernelsizeLow}, Accuracy: {kernelsizeAccuracyLow}")
print(f"Kernel size high: {kernelsizeHigh}, Accuracy: {kernelsizeAccuracyHigh}")

plt.plot(kernelsizeRange,kernelsizeAccuracy)


##get convolution layer accuracies
convolutionlayerRange = (1,4)
convolutionlayerRange = np.arange(min(convolutionlayerRange),max(convolutionlayerRange)+1,1)
convolutionlayerAccuracy = []

kernel_size_Conv2D, number_of_convolutional_layers, number_of_filter_conv2D, number_of_neurons, learning_rate, number_of_epochs, batch_size, momentum = resetParameters()

for number_of_convolutional_layers in convolutionlayerRange:
  val_accuracy = GetValAccuracy(kernel_size_Conv2D,number_of_convolutional_layers,number_of_filter_conv2D,number_of_neurons,learning_rate,number_of_epochs,batch_size,momentum)
  convolutionlayerAccuracy.append(val_accuracy)

i = 0
for accuracy in convolutionlayerAccuracy:
  if accuracy == min(convolutionlayerAccuracy):
    convolutionlayerLow = convolutionlayerRange[i]
    convolutionlayerAccuracyLow = accuracy
  if accuracy == max(convolutionlayerAccuracy):
    convolutionlayerHigh = convolutionlayerRange[i]
    convolutionlayerAccuracyHigh = accuracy 
  i += 1

print(f"Layers low: {convolutionlayerLow}, Accuracy: {convolutionlayerAccuracyLow}")
print(f"Layers high: {convolutionlayerHigh}, Accuracy: {convolutionlayerAccuracyHigh}")


##get filter accuracies
filterRange = [16,32,64,128,264]
filterAccuracy = []

kernel_size_Conv2D, number_of_convolutional_layers, number_of_filter_conv2D, number_of_neurons, learning_rate, number_of_epochs, batch_size, momentum = resetParameters()

for number_of_filter_conv2D in filterRange:
  val_accuracy = GetValAccuracy(kernel_size_Conv2D,number_of_convolutional_layers,number_of_filter_conv2D,number_of_neurons,learning_rate,number_of_epochs,batch_size,momentum)
  filterAccuracy.append(val_accuracy)

i = 0
for accuracy in filterAccuracy:
  if accuracy == min(filterAccuracy):
    filterLow = filterRange[i]
    filterAccuracyLow = accuracy
  if accuracy == max(filterAccuracy):
    filterHigh = filterRange[i]
    filterAccuracyHigh = accuracy 
  i += 1

print(f"Filters low: {filterLow}, Accuracy: {filterAccuracyLow}")
print(f"Filters high: {filterHigh}, Accuracy: {filterAccuracyHigh}")


##get neuron accuracies
neuronRange = (50,300)
neuronRange = np.arange(min(neuronRange),max(neuronRange)+1,25)
neuronAccuracy = []

kernel_size_Conv2D, number_of_convolutional_layers, number_of_filter_conv2D, number_of_neurons, learning_rate, number_of_epochs, batch_size, momentum = resetParameters()

for number_of_neurons in neuronRange:
  val_accuracy = GetValAccuracy(kernel_size_Conv2D,number_of_convolutional_layers,number_of_filter_conv2D,number_of_neurons,learning_rate,number_of_epochs,batch_size,momentum)
  neuronAccuracy.append(val_accuracy)

i = 0
for accuracy in neuronAccuracy:
  if accuracy == min(neuronAccuracy):
    neuronLow = neuronRange[i]
    neuronAccuracyLow = accuracy
  if accuracy == max(neuronAccuracy):
    neuronHigh = neuronRange[i]
    neuronAccuracyHigh = accuracy 
  i += 1

print(f"Neurons low: {neuronLow}, Accuracy: {neuronAccuracyLow}")
print(f"Neurons size high: {neuronHigh}, Accuracy: {neuronAccuracyHigh}")


##get learning rate accuracies
learningrateRange = [0.0001,0.001,0.01,0.1,1]
#learningrateRange = np.arange(min(learningrateRange),max(learningrateRange)+0.0001,0.1)   #Caution: this range is very large
learningrateAccuracy = []

kernel_size_Conv2D, number_of_convolutional_layers, number_of_filter_conv2D, number_of_neurons, learning_rate, number_of_epochs, batch_size, momentum = resetParameters()

for learning_rate in learningrateRange:
  val_accuracy = GetValAccuracy(kernel_size_Conv2D,number_of_convolutional_layers,number_of_filter_conv2D,number_of_neurons,learning_rate,number_of_epochs,batch_size,momentum)
  learningrateAccuracy.append(val_accuracy)

i = 0
for accuracy in learningrateAccuracy:
  if accuracy == min(learningrateAccuracy):
    learningrateLow = learningrateRange[i]
    learningrateAccuracyLow = accuracy
  if accuracy == max(learningrateAccuracy):
    learningrateHigh = learningrateRange[i]
    learningrateAccuracyHigh = accuracy 
  i += 1

print(f"Learning rate low: {learningrateLow}, Accuracy: {learningrateAccuracyLow}")
print(f"Learning rate high: {learningrateHigh}, Accuracy: {learningrateAccuracyHigh}")


##get batch size accuracies
batchsizeRange = (10,100)
batchsizeRange = np.arange(min(batchsizeRange),max(batchsizeRange)+1,10)
batchsizeAccuracy = []

kernel_size_Conv2D, number_of_convolutional_layers, number_of_filter_conv2D, number_of_neurons, learning_rate, number_of_epochs, batch_size, momentum = resetParameters()

for batch_size in batchsizeRange:
  val_accuracy = GetValAccuracy(kernel_size_Conv2D,number_of_convolutional_layers,number_of_filter_conv2D,number_of_neurons,learning_rate,number_of_epochs,batch_size,momentum)
  batchsizeAccuracy.append(val_accuracy)

i = 0
for accuracy in batchsizeAccuracy:
  if accuracy == min(batchsizeAccuracy):
    batchsizeLow = batchsizeRange[i]
    batchsizeAccuracyLow = accuracy
  if accuracy == max(batchsizeAccuracy):
    batchsizeHigh = batchsizeRange[i]
    batchsizeAccuracyHigh = accuracy 
  i += 1

print(f"Kernel size low: {batchsizeLow}, Accuracy: {batchsizeAccuracyLow}")
print(f"Kernel size high: {batchsizeHigh}, Accuracy: {batchsizeAccuracyHigh}")


##get momentum accuracies
momentumRange = (0.5,0.99)
momentumRange = np.arange(min(momentumRange),max(momentumRange)+0.01,0.05)
momentumAccuracy = []

kernel_size_Conv2D, number_of_convolutional_layers, number_of_filter_conv2D, number_of_neurons, learning_rate, number_of_epochs, batch_size, momentum = resetParameters()

for momentum in momentumRange:
  val_accuracy = GetValAccuracy(kernel_size_Conv2D,number_of_convolutional_layers,number_of_filter_conv2D,number_of_neurons,learning_rate,number_of_epochs,batch_size,momentum)
  momentumAccuracy.append(val_accuracy)

i = 0
for accuracy in momentumAccuracy:
  if accuracy == min(momentumAccuracy):
    momentumLow = momentumRange[i]
    momentumAccuracyLow = accuracy
  if accuracy == max(momentumAccuracy):
    momentumHigh = momentumRange[i]
    momentumAccuracyHigh = accuracy 
  i += 1

print(f"Momentum low: {momentumLow}, Accuracy: {momentumAccuracyLow}")
print(f"Momentum size high: {momentumHigh}, Accuracy: {momentumAccuracyHigh}")


##get epoch accuracies
epochRange = (1,40)
epochRange = np.arange(min(epochRange),max(epochRange)+1,5)
epochAccuracy = []

kernel_size_Conv2D, number_of_convolutional_layers, number_of_filter_conv2D, number_of_neurons, learning_rate, number_of_epochs, batch_size, momentum = resetParameters()

for number_of_epochs in epochRange:
  val_accuracy = GetValAccuracy(kernel_size_Conv2D,number_of_convolutional_layers,number_of_filter_conv2D,number_of_neurons,learning_rate,number_of_epochs,batch_size,momentum)
  epochAccuracy.append(val_accuracy)

i = 0
for accuracy in epochAccuracy:
  if accuracy == min(epochAccuracy):
    epochLow = epochRange[i]
    epochAccuracyLow = accuracy
  if accuracy == max(epochAccuracy):
    epochHigh = epochRange[i]
    epochAccuracyHigh = accuracy 
  i += 1

print(f"Kernel size low: {epochLow}, Accuracy: {epochAccuracyLow}")
print(f"Kernel size high: {epochHigh}, Accuracy: {epochAccuracyHigh}")


###Run experiment
##Set parameter values
#lows = (min(kernelsizeLow, kernelsizeHigh),min(convolutionlayerLow, convolutionlayerHigh),min(filterLow, filterHigh),min(neuronLow, neuronHigh),min(learningrateLow, learningrateHigh),min(epochLow, epochHigh),min(batchsizeLow, batchsizeHigh),min(momentumLow, momentumHigh))
#highs = (max(kernelsizeLow, kernelsizeHigh),max(convolutionlayerLow, convolutionlayerHigh),max(filterLow, filterHigh),max(neuronLow, neuronHigh),max(learningrateLow, learningrateHigh),max(epochLow, epochHigh),max(batchsizeLow, batchsizeHigh),max(momentumLow, momentumHigh))
lows = (1,2,16,50,0.001,1,10,0.5)   #Notice: These are for testing. You might want to use the lines above
highs = (3,4,128,125,0.9,36,100,0.95)   #These are for testing. You might want to use the lines above
parameters = (lows,highs)

#Set subsets
subsets = [[], ["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"], ["h"], ["a", "b"], ["a", "c"], ["a", "d"], ["a", "e"], ["a", "f"], ["a", "g"], ["a", "h"], ["b", "c"], ["b", "d"], ["b", "e"], ["b", "f"], ["b", "g"], ["b", "h"], ["c", "d"], ["c", "e"], ["c", "f"], ["c", "g"], ["c", "h"], ["d", "e"], ["d", "f"], ["d", "g"], ["d", "h"], ["e", "f"], ["e", "g"], ["e", "h"], ["f", "g"], ["f", "h"], ["g", "h"], ["a", "b", "c"], ["a", "b", "d"], ["a", "b", "e"], ["a", "b", "f"], ["a", "b", "g"], ["a", "b", "h"], ["a", "c", "d"], ["a", "c", "e"], ["a", "c", "f"], ["a", "c", "g"], ["a", "c", "h"], ["a", "d", "e"], ["a", "d", "f"], ["a", "d", "g"], ["a", "d", "h"], ["a", "e", "f"], ["a", "e", "g"], ["a", "e", "h"], ["a", "f", "g"], ["a", "f", "h"], ["a", "g", "h"], ["b", "c", "d"], ["b", "c", "e"], ["b", "c", "f"], ["b", "c", "g"], ["b", "c", "h"], ["b", "d", "e"], ["b", "d", "f"], ["b", "d", "g"], ["b", "d", "h"], ["b", "e", "f"], ["b", "e", "g"], ["b", "e", "h"], ["b", "f", "g"], ["b", "f", "h"], ["b", "g", "h"], ["c", "d", "e"], ["c", "d", "f"], ["c", "d", "g"], ["c", "d", "h"], ["c", "e", "f"], ["c", "e", "g"], ["c", "e", "h"], ["c", "f", "g"], ["c", "f", "h"], ["c", "g", "h"], ["d", "e", "f"], ["d", "e", "g"], ["d", "e", "h"], ["d", "f", "g"], ["d", "f", "h"], ["d", "g", "h"], ["e", "f", "g"], ["e", "f", "h"], ["e", "g", "h"], ["f", "g", "h"], ["a", "b", "c", "d"], ["a", "b", "c", "e"], ["a", "b", "c", "f"], ["a", "b", "c", "g"], ["a", "b", "c", "h"], ["a", "b", "d", "e"], ["a", "b", "d", "f"], ["a", "b", "d", "g"], ["a", "b", "d", "h"], ["a", "b", "e", "f"], ["a", "b", "e", "g"], ["a", "b", "e", "h"], ["a", "b", "f", "g"], ["a", "b", "f", "h"], ["a", "b", "g", "h"], ["a", "c", "d", "e"], ["a", "c", "d", "f"], ["a", "c", "d", "g"], ["a", "c", "d", "h"], ["a", "c", "e", "f"], ["a", "c", "e", "g"], ["a", "c", "e", "h"], ["a", "c", "f", "g"], ["a", "c", "f", "h"], ["a", "c", "g", "h"], ["a", "d", "e", "f"], ["a", "d", "e", "g"], ["a", "d", "e", "h"], ["a", "d", "f", "g"], ["a", "d", "f", "h"], ["a", "d", "g", "h"], ["a", "e", "f", "g"], ["a", "e", "f", "h"], ["a", "e", "g", "h"], ["a", "f", "g", "h"], ["b", "c", "d", "e"], ["b", "c", "d", "f"], ["b", "c", "d", "g"], ["b", "c", "d", "h"], ["b", "c", "e", "f"], ["b", "c", "e", "g"], ["b", "c", "e", "h"], ["b", "c", "f", "g"], ["b", "c", "f", "h"], ["b", "c", "g", "h"], ["b", "d", "e", "f"], ["b", "d", "e", "g"], ["b", "d", "e", "h"], ["b", "d", "f", "g"], ["b", "d", "f", "h"], ["b", "d", "g", "h"], ["b", "e", "f", "g"], ["b", "e", "f", "h"], ["b", "e", "g", "h"], ["b", "f", "g", "h"], ["c", "d", "e", "f"], ["c", "d", "e", "g"], ["c", "d", "e", "h"], ["c", "d", "f", "g"], ["c", "d", "f", "h"], ["c", "d", "g", "h"], ["c", "e", "f", "g"], ["c", "e", "f", "h"], ["c", "e", "g", "h"], ["c", "f", "g", "h"], ["d", "e", "f", "g"], ["d", "e", "f", "h"], ["d", "e", "g", "h"], ["d", "f", "g", "h"], ["e", "f", "g", "h"], ["a", "b", "c", "d", "e"], ["a", "b", "c", "d", "f"], ["a", "b", "c", "d", "g"], ["a", "b", "c", "d", "h"], ["a", "b", "c", "e", "f"], ["a", "b", "c", "e", "g"], ["a", "b", "c", "e", "h"], ["a", "b", "c", "f", "g"], ["a", "b", "c", "f", "h"], ["a", "b", "c", "g", "h"], ["a", "b", "d", "e", "f"], ["a", "b", "d", "e", "g"], ["a", "b", "d", "e", "h"], ["a", "b", "d", "f", "g"], ["a", "b", "d", "f", "h"], ["a", "b", "d", "g", "h"], ["a", "b", "e", "f", "g"], ["a", "b", "e", "f", "h"], ["a", "b", "e", "g", "h"], ["a", "b", "f", "g", "h"], ["a", "c", "d", "e", "f"], ["a", "c", "d", "e", "g"], ["a", "c", "d", "e", "h"], ["a", "c", "d", "f", "g"], ["a", "c", "d", "f", "h"], ["a", "c", "d", "g", "h"], ["a", "c", "e", "f", "g"], ["a", "c", "e", "f", "h"], ["a", "c", "e", "g", "h"], ["a", "c", "f", "g", "h"], ["a", "d", "e", "f", "g"], ["a", "d", "e", "f", "h"], ["a", "d", "e", "g", "h"], ["a", "d", "f", "g", "h"], ["a", "e", "f", "g", "h"], ["b", "c", "d", "e", "f"], ["b", "c", "d", "e", "g"], ["b", "c", "d", "e", "h"], ["b", "c", "d", "f", "g"], ["b", "c", "d", "f", "h"], ["b", "c", "d", "g", "h"], ["b", "c", "e", "f", "g"], ["b", "c", "e", "f", "h"], ["b", "c", "e", "g", "h"], ["b", "c", "f", "g", "h"], ["b", "d", "e", "f", "g"], ["b", "d", "e", "f", "h"], ["b", "d", "e", "g", "h"], ["b", "d", "f", "g", "h"], ["b", "e", "f", "g", "h"], ["c", "d", "e", "f", "g"], ["c", "d", "e", "f", "h"], ["c", "d", "e", "g", "h"], ["c", "d", "f", "g", "h"], ["c", "e", "f", "g", "h"], ["d", "e", "f", "g", "h"], ["a", "b", "c", "d", "e", "f"], ["a", "b", "c", "d", "e", "g"], ["a", "b", "c", "d", "e", "h"], ["a", "b", "c", "d", "f", "g"], ["a", "b", "c", "d", "f", "h"], ["a", "b", "c", "d", "g", "h"], ["a", "b", "c", "e", "f", "g"], ["a", "b", "c", "e", "f", "h"], ["a", "b", "c", "e", "g", "h"], ["a", "b", "c", "f", "g", "h"], ["a", "b", "d", "e", "f", "g"], ["a", "b", "d", "e", "f", "h"], ["a", "b", "d", "e", "g", "h"], ["a", "b", "d", "f", "g", "h"], ["a", "b", "e", "f", "g", "h"], ["a", "c", "d", "e", "f", "g"], ["a", "c", "d", "e", "f", "h"], ["a", "c", "d", "e", "g", "h"], ["a", "c", "d", "f", "g", "h"], ["a", "c", "e", "f", "g", "h"], ["a", "d", "e", "f", "g", "h"], ["b", "c", "d", "e", "f", "g"], ["b", "c", "d", "e", "f", "h"], ["b", "c", "d", "e", "g", "h"], ["b", "c", "d", "f", "g", "h"], ["b", "c", "e", "f", "g", "h"], ["b", "d", "e", "f", "g", "h"], ["c", "d", "e", "f", "g", "h"], ["a", "b", "c", "d", "e", "f", "g"], ["a", "b", "c", "d", "e", "f", "h"], ["a", "b", "c", "d", "e", "g", "h"], ["a", "b", "c", "d", "f", "g", "h"], ["a", "b", "c", "e", "f", "g", "h"], ["a", "b", "d", "e", "f", "g", "h"], ["a", "c", "d", "e", "f", "g", "h"], ["b", "c", "d", "e", "f", "g", "h"], ["a", "b", "c", "d", "e", "f", "g", "h"]]

accuracyList = []
i = 0
j = 0
for subset in subsets:
    #get kernel size level
    if ('a' in subset) == True: a = 1 
    else: a = 0
    
    #get convolution layers level
    if ('b' in subset) == True: b = 1 
    else: b = 0
    
    #get number of filters level
    if ('c' in subset) == True: c = 1 
    else: c = 0
    
    #get number of neurons level
    if ('d' in subset) == True: d = 1 
    else: d = 0
    
    #get learning rate level
    if ('e' in subset) == True: e = 1 
    else: e = 0
    
    #get number of epochs level
    if ('f' in subset) == True: f = 1 
    else: f = 0
    
    #get batch size level
    if ('g' in subset) == True: g = 1 
    else: g = 0
        
    #get momentum level
    if ('h' in subset) == True: h = 1 
    else: h = 0
        
    #assign levels    
    kernel_size_Conv2D,number_of_convolutional_layers,number_of_filter_conv2D,number_of_neurons,learning_rate,number_of_epochs,batch_size,momentum = parameters[a][0],parameters[b][1],parameters[c][2],parameters[d][3],parameters[e][4],parameters[f][5],parameters[g][6],parameters[h][7]

    
    if f==0 and g==1: #Different combinations of f and g were used to run the experiment in batches. You might not need this on your machine (good luck)
    ##if i==18:
      j+=1
      print(f"Running experiment {i} ({j}/64): combination ({a},{b},{c},{d},{e},{f},{g},{h})")
      print(f"Levels: {kernel_size_Conv2D}, {number_of_convolutional_layers}, {number_of_filter_conv2D}, {number_of_neurons}, {learning_rate}, {number_of_epochs}, {batch_size}, {momentum}")
      #calculate accuracies and add to list
      val_accuracy = GetValAccuracy(kernel_size_Conv2D,number_of_convolutional_layers,number_of_filter_conv2D,number_of_neurons,learning_rate,number_of_epochs,batch_size,momentum)
      print("")
      accuracyList.append((i,val_accuracy))
    #accuracyList.append((i,(a,b,c,d,e,f,g,h)))  
    
    i+=1

##Return the accuracy list. This is the manual step. Copy the values into a spreadsheet and run the next batch
print(accuracyList)