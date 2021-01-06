from __future__ import print_function
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from sklearn.metrics import confusion_matrix
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd     # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from time import time 
import os
import cv2                  # OpenCV
#from keras.datasets import fashion_mnist
from keras.layers import Conv2D, MaxPooling2D
#from keras.layers.advanced_activations import LeakyReLU
print('Loaded required libraries')

'''
Load dataset and pre-processing data
------------------------------------
'''
from keras.datasets import cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print('Dowloaded CIFAR-10 dataset')

print(train_images.shape)       # (50000, 32, 32, 3) 
print(train_labels.shape)       # (50000, 1) 
print(test_images.shape)        # (10000, 32, 32, 3)
print(test_labels.shape)        # (10000, 1) 

train_labels = train_labels.reshape(len(train_labels))  # (50000,)
test_labels = test_labels.reshape(len(test_labels))     # (10000,)

categories = np.unique(train_labels)
print('# of classes: {nclass}'.format(nclass=len(categories)))

# Resize the images 48*48 as required by VGG16
from keras.preprocessing.image import img_to_array, array_to_img
train_images = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in train_images])
test_images = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in test_images])
print(train_images.shape, test_images.shape)

# Normalise the data and change data type
train_images = train_images / 255.
test_images = test_images / 255.
train_images = train_images.astype('float32')
test_images = test_images.astype('float32') 

# # Converting Labels to one hot encoded format
# from keras.utils import to_categorical
# train_Y_one_hot = to_categorical(train_Y)
# test_Y_one_hot = to_categorical(test_Y)

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
img_idx = np.random.randint(len(train_images))
plt.imshow(train_images[img_idx], cmap='gray')
plt.title("(Label: " + str(label_dict[train_labels[img_idx]]) + ")")
# Display a random image in the testing data
plt.subplot(122)
img_idx = np.random.randint(len(test_images))
plt.imshow(test_images[img_idx], cmap='gray')
plt.title("(Label: " + str(label_dict[test_labels[img_idx]]) + ")")
plt.show()

# Preprocessing the input 
from keras.applications.vgg16 import preprocess_input
X_train = preprocess_input(train_images)
X_test  = preprocess_input(test_images)
y_train = train_labels
y_test = test_labels

print('Loaded and pre-processed the dataset')

'''
Load the VGG16 model 
Create a new model based on VGG16
Train the new model
Save trained model to JSON + HDF5 files
Save the history of training to a *.npy file 
------------------------------------
'''
# Define the parameters for instanitaing VGG16 model. 
IMG_WIDTH = 48
IMG_HEIGHT = 48
IMG_DEPTH = 3
BATCH_SIZE = 64

# load model without classifier layers
base_model = VGG16(
    weights='imagenet',       # load weights pretrained on ImageNet
    input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH),
    include_top=False      # do not include the ImageNet classifier at the top
    )
print('Loaded the base model')
# print('base model:')
# base_model.summary()

# create a new model on top 
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024,activation='relu')(x)   # dense layer #1
# x = Dense(1024,activation='relu')(x)   # dense layer #2
x = Dense(512,activation='relu')(x)    # dense layer #3
predictions = Dense(len(categories),activation='softmax')(x)    # final layer with softmax activation 
model = Model(inputs=base_model.input,outputs=predictions)      # specify the input and output of the new model

# mark trainable/non-trainable layers
for layer in model.layers[:len(base_model.layers)]:
    layer.trainable = False
for layer in model.layers[len(base_model.layers):]:
    layer.trainable = True 
model.summary()

# complile the new model 
model.compile(
                optimizer='Adam',
                # optimizer=keras.optimizers.Adam(),
                # loss='categorical_crossentropy',        # loss function 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])                   # evaluation metric 
print('Compiled the new model')

# prepare data for training 
from sklearn.model_selection import train_test_split
train_X,valid_X,train_label,valid_label = train_test_split(X_train,y_train,test_size=0.2,random_state=13)

# train the new model 
N_EPOCHS = 20
BATCH_SIZE = 64
history = model.fit(train_X, train_label, 
                    epochs=N_EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    validation_data=(valid_X,valid_label)
                    )

# serialize the model to JSON 
model_json = model.to_json()
with open("output/vgg16_cifar10.json", "w") as json_file: 
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("output/vgg16_cifar10.h5")
print("Saved model to disk (JSON + HDF5 files)")

# save the training history to a npy file
np.save('output/vgg16_cifar10_history.npy',history.history)
print('Saved training history to a npy file')

# plot figures for training history 
accuracy = history.history['accuracy'] 
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(accuracy))
N_EPOCHS2 = len(accuracy)

plt.figure
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xticks(np.arange(0,N_EPOCHS2+1,N_EPOCHS2/10))
plt.title('Training and validation accuracy')
plt.legend()
plt.grid(True)
plt.figure()

plt.figure
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xticks(np.arange(0,N_EPOCHS2+1,N_EPOCHS2/10))
plt.title('Training and validation loss')
plt.legend()
plt.grid(True)
plt.show()

'''
Load training history from files and plot figures 
--------------------------------
''' 
# # load the training history from npy files 
# history1=np.load('output/vgg16_cifar10_history_p1.npy',allow_pickle='TRUE').item()
# history2=np.load('output/vgg16_cifar10_history_p2.npy',allow_pickle='TRUE').item()
# history3=np.load('output/vgg16_cifar10_history.npy',allow_pickle='TRUE').item()

# # list all data in history 
# # print(history.history.keys())
# print(history.keys())

# # performance of the classifier model in tranning
# accuracy = history1['accuracy'] + history2['accuracy'] + history3['accuracy']
# val_accuracy = history1['val_accuracy'] + history2['val_accuracy'] + history3['val_accuracy']
# loss = history1['loss'] + history2['loss'] + history3['loss']
# val_loss = history1['val_loss'] + history2['val_loss'] + history3['val_loss']
# epochs = range(len(accuracy))
# N_EPOCHS2 = len(accuracy)

# plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
# plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
# plt.xlabel('epoch')
# plt.ylabel('accuracy')
# plt.xticks(np.arange(0,N_EPOCHS2+1,N_EPOCHS2/10))
# plt.title('Training and validation accuracy')
# plt.legend()
# plt.grid(True)
# plt.figure()

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.xlabel('epoch')
# plt.ylabel('loss')
# plt.xticks(np.arange(0,N_EPOCHS2+1,N_EPOCHS2/10))
# plt.title('Training and validation loss')
# plt.legend()
# plt.grid(True)
# plt.show()

'''
Load the trained model from files
--------------------------------
'''
# from keras.models import model_from_json

# # load the model 
# fpath_json = 'output/vgg16_cifar10.json'
# json_file = open(fpath_json, "r")
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)

# # load weights into the loaded model 
# fpath_h5 = 'output/vgg16_cifar10.h5'
# model.load_weights(fpath_h5)
# print("Loaded model from disk")

# model.summary()

# # complile the new model 
# model.compile(
#                 optimizer='Adam',
#                 # optimizer=keras.optimizers.Adam(),
#                 # loss='categorical_crossentropy',        # loss function 
#                 loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy'])                   # evaluation metric 
# print('Loaded and compiled the new model')

'''
Evaludate the model on the testing set
Confusion matrix & related KPIs
----------------------------------
'''
# Metrics to evaluate accuracy and loss in test dataset
t1 = time()
loss, acc = model.evaluate(X_test, y_test)
test_time = time() - t1
print('Loss = {loss:.4f}\nAccuracy = {acc:.4f}'.format(loss=loss,acc=acc))
print('Test time : {t:.4f} sec'.format(t=test_time))

# get predictions of the model 
y_pred_onehot = model.predict(X_test)
y_pred = np.array([np.argmax(y_pred_onehot[i]) for i in range(len(y_pred_onehot))])

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

# plot figure 
labels = np.array([label_dict[i] for i in categories])
cm_analysis(y_test, y_pred, labels, ymap=None, figsize=(14,12), title="VGG16 (transfer learning), CIFAR-10 dataset")

# other KPIs
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score, classification_report
print('Accuracy score : {x:.4f}'.format(x=accuracy_score(y_test,y_pred)))
print('F1 score : {x:.4f}'.format(x=f1_score(y_test,y_pred,average='macro'))) 
print('Precision score : {x:.4f}'.format(x=precision_score(y_test,y_pred,average='macro')))
print('Recall score : {x:.4f}'.format(x=recall_score(y_test,y_pred,average='macro')))
