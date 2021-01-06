from __future__ import print_function
from matplotlib import pyplot as plt
import gzip
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from time import time 
import os 
import cv2                  # OpenCV
print('Loaded all required libraries')

'''
Load dataset and pre-processing data
------------------------------------
'''
train_data = pd.read_csv('input/fashion-mnist/fashion-mnist_train.csv')
test_data = pd.read_csv('input/fashion-mnist/fashion-mnist_test.csv')

# print(train_data.shape)        #(60,000*785)
# print(test_data.shape)         #(10000,785)
# train_data.head() 

train_X = np.array(train_data.iloc[:,1:])
test_X = np.array(test_data.iloc[:,1:])
y_train = np.array (train_data.iloc[:,0])   # (60000,)
y_test = np.array(test_data.iloc[:,0])      # (10000,)

categories = np.unique(y_train)
print('# of classes: {nclass}'.format(nclass=len(categories)))

# # Convert grey images (1 channel) to 3-channel images
# train_X=np.dstack([train_X]*3)
# test_X=np.dstack([test_X]*3)
# # print(train_X.shape,test_X.shape)       # ((60000, 784, 3), (10000, 784, 3))

# Reshape images as per the tensor format required by tensorflow
train_X = train_X.reshape(-1, 28,28,1)
test_X= test_X.reshape (-1,28,28,1)
print(train_X.shape,test_X.shape)       # ((60000, 28, 28, 1), (10000, 28, 28, 1))

# # Resize the images 48*48 as required by VGG16
# IMG_SIZE = 48; 
# from keras.preprocessing.image import img_to_array, array_to_img
# train_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in train_X])
# test_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in test_X])
# print(train_X.shape, test_X.shape)

# Normalise the data and change data type
train_X = train_X / 255.
test_X = test_X / 255.
train_X = train_X.astype('float32')
test_X = test_X.astype('float32') 

# # Converting Labels to one hot encoded format
# from keras.utils import to_categorical
# train_Y_one_hot = to_categorical(train_Y)
# test_Y_one_hot = to_categorical(test_Y)

# Preprocessing the input 
# X_train = preprocess_input(train_X)
# X_test  = preprocess_input(test_X)
X_train = train_X
X_test  = test_X

# Create dictionary of target classes
label_dict = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot',
}

plt.figure(figsize=[5,5])

# Display a random image in the training set
plt.subplot(121)
img_idx = np.random.randint(len(X_train))
plt.imshow(X_train[img_idx], cmap='gray')
plt.title("(Label: " + str(label_dict[y_train[img_idx]]) + ")")

# Display a random image in the testing data
plt.subplot(122)
img_idx = np.random.randint(len(X_test))
plt.imshow(X_test[img_idx], cmap='gray')
plt.title("(Label: " + str(label_dict[y_test[img_idx]]) + ")")

'''
Generate and train the convolutional autoencoder 
------------------------------------
'''
# Split the training set into 2 parts: train_X and valid_X
# Used for training the autoencoder -> X_train are passed twice
from sklearn.model_selection import train_test_split
train_X,valid_X,train_ground,valid_ground = train_test_split(X_train,
                                                             X_train,
                                                             test_size=0.2,
                                                             random_state=13)

# parameters for training the autoencoder 
batch_size = 64
N_EPOCHS = 20
inChannel = 1
x, y = 28, 28
input_img = Input(shape = (x, y, inChannel))
num_classes = len(categories)

# define the encoder
def encoder(input_img):
    #encoder
    #input = 28 x 28 x 1 (wide and thin)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img) #28 x 28 x 32
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) #14 x 14 x 32
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) #14 x 14 x 64
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) #7 x 7 x 64
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) #7 x 7 x 128 (small and thick)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv3) #7 x 7 x 256 (small and thick)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    conv4 = BatchNormalization()(conv4)
    return conv4

# define the decoder 
def decoder(conv4):    
    #decoder
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4) #7 x 7 x 128
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5) #7 x 7 x 64
    conv6 = BatchNormalization()(conv6)
    conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)
    conv6 = BatchNormalization()(conv6)
    up1 = UpSampling2D((2,2))(conv6) #14 x 14 x 64
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up1) # 14 x 14 x 32
    conv7 = BatchNormalization()(conv7)
    conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)
    conv7 = BatchNormalization()(conv7)
    up2 = UpSampling2D((2,2))(conv7) # 28 x 28 x 32
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(up2) # 28 x 28 x 1
    return decoded

# compile the model 
autoencoder = Model(input_img, decoder(encoder(input_img)))
autoencoder.compile(loss='mean_squared_error', optimizer = RMSprop())
autoencoder.summary()

# train the model 
autoencoder_train = autoencoder.fit(train_X, train_ground, 
            batch_size=batch_size,
            epochs=N_EPOCHS,
            verbose=1,
            validation_data=(valid_X, valid_ground))

# visualize the model performance
loss = autoencoder_train.history['loss']
val_loss = autoencoder_train.history['val_loss']
epochs = range(N_EPOCHS)
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xticks(np.arange(0,N_EPOCHS,2))
plt.title('Training and validation loss')
plt.legend()
plt.grid(True)
plt.show()

# serialize the model to JSON 
model_json = autoencoder.to_json()
with open("output/autoencoder_fashion_mnist.json", "w") as json_file: 
    json_file.write(model_json)

# serialize weights to HDF5
autoencoder.save_weights("output/autoencoder_fashion_mnist.h5")
print("Saved model to disk (JSON + HDF5 files)")

# save the training history to a npy file
np.save('output/autoencoder_fashion_mnist_history.npy',autoencoder_train.history)
print('saved training history to a npy file')

'''
Autoencoder as feature extractor + MLP
-----------------------------------------
'''
# convert the labels to one-hot encoding vectors
train_Y_one_hot = to_categorical(y_train)
test_Y_one_hot = to_categorical(y_test)
train_X,valid_X,train_label,valid_label = train_test_split(X_train,train_Y_one_hot,test_size=0.2,random_state=13)
# train_X.shape,valid_X.shape,train_label.shape,valid_label.shape # ((48000, 28, 28, 1), (12000, 28, 28, 1), (48000, 10), (12000, 10))

# define the classifier 
# the encoder part is the same as we did for the autoencoder 
# the fully-connected layers will be stacked up with the encoder (of the autoencoder)
def fc(enco):
    flat = Flatten()(enco)
    den = Dense(128, activation='relu')(flat)
    out = Dense(num_classes, activation='softmax')(den)
    return out

encode = encoder(input_img)
encoder_model = Model(input_img,encoder(input_img))
num_layers = len(encoder_model.layers)
classifier_model = Model(input_img,fc(encode))

for l1,l2 in zip(classifier_model.layers[:num_layers],autoencoder.layers[0:num_layers]):
    l1.set_weights(l2.get_weights())

# check whether the weights of the encoder part of the classifier model are the same as those of the autoencoder
np.array_equal(autoencoder.get_weights()[0][1],classifier_model.get_weights()[0][1])

# make layers for the encoder part non-trainable 
# (the encoder part is already trained)
for layer in classifier_model.layers[:num_layers]:
    layer.trainable = False

# compile the model 
classifier_model.compile(loss='categorical_crossentropy', 
                    optimizer='Adam',
                    metrics=['accuracy'])

# summary of the classifier model 
classifier_model.summary()

# train the classifier model 
N_EPOCHS2 = 10
classify_train = classifier_model.fit(train_X, train_label, 
                                batch_size=64,
                                epochs=N_EPOCHS2,
                                verbose=1,
                                validation_data=(valid_X, valid_label))

# serialize the model to JSON 
model_json = classifier_model.to_json()
with open("output/autoencoder_mlp_fashion_mnist.json", "w") as json_file: 
    json_file.write(model_json)

# serialize weights to HDF5
classifier_model.save_weights("output/autoencoder_mlp_fashion_mnist.h5")
print("Saved model to disk (JSON + HDF5 files)")

# save the training history to a npy file
np.save('output/autoencoder_mlp_fashion_mnist_history.npy',classify_train.history)
print('saved training history to a npy file')

# performance of the classifier model when tranning
accuracy = classify_train.history['accuracy']
val_accuracy = classify_train.history['val_accuracy']
loss = classify_train.history['loss']
val_loss = classify_train.history['val_loss']
epochs = range(len(accuracy))

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xticks(np.arange(0,N_EPOCHS2,2))
plt.title('Training and validation accuracy')
plt.legend()
plt.grid(True)
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xticks(np.arange(0,N_EPOCHS2,2))
plt.title('Training and validation loss')
plt.legend()
plt.grid(True)
plt.show()

# performance on the test set
t1 = time()
loss, accuracy = classifier_model.evaluate(X_test, test_Y_one_hot)
test_time = time() - t1
print('Loss = {loss:.4f}\nAccuracy = {acc:.4f}'.format(loss=loss,acc=accuracy))
print('Test time : {t:.4f} sec'.format(t=test_time))