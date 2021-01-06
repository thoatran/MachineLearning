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
Load dataset and pre-process data
------------------------------------
'''
from keras.datasets import cifar10
(train_X, train_labels), (test_X, test_labels) = cifar10.load_data()
print('Dowloaded CIFAR-10 dataset')

print(train_X.shape)       # (50000, 32, 32, 3) 
print(train_labels.shape)       # (50000, 1) 
print(test_X.shape)        # (10000, 32, 32, 3)
print(test_labels.shape)        # (10000, 1) 

train_labels = train_labels.reshape(len(train_labels))  # (50000,)
test_labels = test_labels.reshape(len(test_labels))     # (10000,)

categories = np.unique(train_labels)
print('# of classes: {nclass}'.format(nclass=len(categories)))

# Reshape images as per the tensor format required by tensorflow
IMG_SIZE = 32
N_CHANELS = 3
train_X = train_X.reshape(-1, IMG_SIZE,IMG_SIZE,N_CHANELS)
test_X= test_X.reshape (-1,IMG_SIZE,IMG_SIZE,N_CHANELS)
print(train_X.shape,test_X.shape)       

# Normalise the data and change data type
train_X = train_X / 255.
test_X = test_X / 255.
train_X = train_X.astype('float32')
test_X = test_X.astype('float32') 

X_train = train_X
X_test  = test_X
y_train = train_labels
y_test = test_labels

# Create dictionary of target classes
label_dict = {
 0: 'airplane',
 1: 'automobile',
 2: 'bird',
 3: 'cat',
 4: 'deer',
 5: 'dog',
 6: 'frog',
 7: 'horse',
 8: 'ship',
 9: 'truck',
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
inChannel = 3
x, y = IMG_SIZE, IMG_SIZE
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
with open("output/autoencoder_cifar10.json", "w") as json_file: 
    json_file.write(model_json)

# serialize weights to HDF5
autoencoder.save_weights("output/autoencoder_cifar10.h5")
print("Saved model to disk (JSON + HDF5 files)")

# save the training history to a npy file
np.save('output/autoencoder_cifar10_history.npy',autoencoder_train.history)
print('saved training history to a npy file')

'''
Load the encoder from files 
-----------------------------------------
'''
# from keras.models import model_from_json

# # load the model 
# fpath_json = 'output/autoencoder_cifar10.json'
# json_file = open(fpath_json, "r")
# loaded_model_json = json_file.read()
# json_file.close()
# autoencoder = model_from_json(loaded_model_json)

# # load weights into the loaded model 
# fpath_h5 = 'output/autoencoder_cifar10.h5'
# autoencoder.load_weights(fpath_h5)
# print("Loaded model from disk")
# # model.summary()

# # complile the new model 
# autoencoder.compile(
#                 optimizer='Adam',
#                 # optimizer=keras.optimizers.Adam(),
#                 # loss='categorical_crossentropy',        # loss function 
#                 loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy'])                   # evaluation metric 
# print('Compiled the new model')

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
with open("output/autoencoder_classifier_cifar10.json", "w") as json_file: 
    json_file.write(model_json)
# serialize weights to HDF5
classifier_model.save_weights("output/autoencoder_classifier_cifar10.h5")
print("Saved model to disk (JSON + HDF5 files)")
# save the training history to a npy file
np.save('output/autoencoder_classifier_cifar10_history.npy',classify_train.history)
print('Saved training history to a npy file')

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

'''
Confusion matrix & related KPIs
----------------------------------
'''
# get predictions of the model 
y_pred_onehot = classifier_model.predict(X_test)
y_pred = np.array([np.argmax(y_pred_onehot[i]) for i in range(len(y_pred_onehot))])

# # confution matrix 
# cm = confusion_matrix(y_test,y_pred) 
# cm_df = pd.DataFrame(cm,
#                     index=[label_dict[i] for i in categories],
#                     columns=[label_dict[i] for i in categories])

# plt.figure(figsize=(7, 6))
# sns.heatmap(cm_df, annot=True, fmt='g')
# plt.title("VGG16 (transfer learning), Fashion-MNIT dataset")
# # plt.title('1NN Classifier, Iris dataset \nAccuracy: %.2f%, Running time: %.2f (s)'%(accuracy_avg, rtime_avg))
# plt.ylabel('True label')
# plt.xlabel('Predicted label')
# plt.show()

def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(10,10), title="Confusion matrix"):
    """
    Generate matrix plot of confusion matrix with pretty annotations.
    The plot image is saved to disk.
    args: 
      y_true:    true label of the data, with shape (nsamples,)
      y_pred:    prediction of the data, with shape (nsamples,)
      filename:  filename of figure file to save
      labels:    string array, name the order of class labels in the confusion matrix.
                 use `clf.classes_` if using scikit-learn models.
                 with shape (nclass,).
      ymap:      dict: any -> string, length == nclass.
                 if not None, map the labels & ys to more understandable strings.
                 Caution: original y_true, y_pred and labels must align.
      figsize:   the size of the figure plotted.
    """
    if ymap is not None:
        y_pred = [ymap[yi] for yi in y_pred]
        y_true = [ymap[yi] for yi in y_true]
        labels = [ymap[yi] for yi in labels]
    # cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm = confusion_matrix(y_test,y_pred) 
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%d/%d\n%.1f%%' % (c, s, p)
            # elif c == 0:
            #     annot[i, j] = ''
            else:
                annot[i, j] = '%d\n%.1f%%' % (c,p)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm.index.name = 'True labels'
    cm.columns.name = 'Predicted labels'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=annot, fmt='', ax=ax)
    #plt.savefig(filename)
    plt.title(title)
    plt.show()

labels = np.array([label_dict[i] for i in categories])
cm_analysis(y_test, y_pred, labels, ymap=None, figsize=(14,12), title="Autoencoder-based classifier, CIFAR-10 dataset")

# other KPIs
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
print('Accuracy score : {x:.4f}'.format(x=accuracy_score(y_test,y_pred)))
print('F1 score : {x:.4f}'.format(x=f1_score(y_test,y_pred,average='macro'))) 
print('Precision score : {x:.4f}'.format(x=precision_score(y_test,y_pred,average='macro')))
print('Recall score : {x:.4f}'.format(x=recall_score(y_test,y_pred,average='macro')))