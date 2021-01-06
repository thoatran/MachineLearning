from __future__ import print_function
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, ImageDataGenerator
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Dense, Flatten, GlobalAveragePooling2D, Dropout
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import train_test_split
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
print('Loaded all libraries')

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

# Convert grey images (1 channel) to 3-channel images
train_X=np.dstack([train_X] * 3)
test_X=np.dstack([test_X]*3)
# print(train_X.shape,test_X.shape)       # ((60000, 784, 3), (10000, 784, 3))

# Reshape images as per the tensor format required by tensorflow
train_X = train_X.reshape(-1, 28,28,3)
test_X= test_X.reshape (-1,28,28,3)
# print(train_X.shape,test_X.shape)       # ((60000, 28, 28, 3), (10000, 28, 28, 3))

# Resize the images 48*48 as required by VGG16
from keras.preprocessing.image import img_to_array, array_to_img
train_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in train_X])
test_X = np.asarray([img_to_array(array_to_img(im, scale=False).resize((48,48))) for im in test_X])
#train_x = preprocess_input(x)
print(train_X.shape, test_X.shape)

# Normalise the data and change data type
train_X = train_X / 255.
test_X = test_X / 255.
train_X = train_X.astype('float32')
test_X = test_X.astype('float32') 

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
img_idx = np.random.randint(len(train_X))
plt.imshow(train_X[img_idx], cmap='gray')
plt.title("(Label: " + str(label_dict[y_train[img_idx]]) + ")")
# Display a random image in the testing data
plt.subplot(122)
img_idx = np.random.randint(len(test_X))
plt.imshow(test_X[img_idx], cmap='gray')
plt.title("(Label: " + str(label_dict[y_test[img_idx]]) + ")")

# Preprocessing the input 
X_train = preprocess_input(train_X)
X_test  = preprocess_input(test_X)

print('Loaded and pre-processed the dataset')

'''
Load the VGG16 model 
------------------------------------
'''
# Define the parameters for instanitaing VGG16 model. 
IMG_WIDTH = 48
IMG_HEIGHT = 48
IMG_DEPTH = 3
BATCH_SIZE = 16

# load model without classifier layers
base_model = VGG16(
    weights='imagenet',       # load weights pretrained on ImageNet
    input_shape=(IMG_HEIGHT,IMG_WIDTH,IMG_DEPTH),
    include_top=False      # do not include the ImageNet classifier at the top
    )
print('loaded the base model')
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
# print('\nnew model:')
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

# Metrics to evaluate accuracy and loss in test dataset
t1 = time()
loss, accuracy = model.evaluate(X_test, y_test)
test_time = time() - t1
print('Loss = {loss:.4f}\nAccuracy = {acc:.4f}'.format(loss=loss,acc=accuracy))
print('Test time : {t:.4f} sec'.format(t=test_time))

'''
save the model to JSON + HDF5 file
save the history of training to a *.npy file 
--------------------------------
'''
# serialize the model to JSON 
model_json = model.to_json()
with open("output/vgg16_fashion_mnist.json", "w") as json_file: 
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("output/vgg16_fashion_mnist.h5")
print("Saved model to disk (JSON + HDF5 files)")

# save the training history to a npy file
np.save('output/vgg16_fashion_mnist_history.npy',history.history)
print('saved training history to a npy file')

'''
Plotting figures 
--------------------------------
''' 
# load the training history from npy files 
history=np.load('output/vgg16_fashion_mnist_history.npy',allow_pickle='TRUE').item()

# plot figures for training history 
accuracy = history['accuracy'] 
val_accuracy = history['val_accuracy']
loss = history['loss']
val_loss = history['val_loss']
epochs = range(len(accuracy))
N_EPOCHS2 = len(accuracy)

plt.figure()
plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b', label='Validation accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xticks(np.arange(0,N_EPOCHS2+1,N_EPOCHS2/10))
plt.title('Training and validation accuracy')
plt.legend()
plt.grid(True)
plt.show()

plt.figure()
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
Load json and create model 
Load the training history 
--------------------------------
'''
# from keras.models import model_from_json

# # load the model 
# fpath_json = 'output/vgg16_fashion_mnist.json'
# # fpath_json = fpath_imac + 'model.json'
# json_file = open(fpath_json, "r")
# loaded_model_json = json_file.read()
# json_file.close()
# model = model_from_json(loaded_model_json)

# # load weights into the loaded model 
# fpath_h5 = 'output/vgg16_fashion_mnist.h5'
# # fpath_h5 = fpath_imac + 'model.h5'
# model.load_weights(fpath_h5)
# print("Loaded model from disk")
# # model.summary()

# # complile the new model 
# model.compile(
#                 optimizer='Adam',
#                 # optimizer=keras.optimizers.Adam(),
#                 # loss='categorical_crossentropy',        # loss function 
#                 loss='sparse_categorical_crossentropy',
#                 metrics=['accuracy'])                   # evaluation metric 
# print('Compiled the new model')

# # continue training the model 
# N_EPOCHS = 20
# history = model.fit(X_train, y_train, 
#                     epochs=N_EPOCHS
#                     # , batch_size=10
#                     )

# # Metrics to evaluate accuracy and loss in test dataset
# loss, accuracy = model.evaluate(X_test, y_test)
# print(accuracy,loss)

'''
Confusion matrix & related KPIs
----------------------------------
'''
# get predictions of the model 
y_pred_onehot = model.predict(X_test)
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
cm_analysis(y_test, y_pred, labels, ymap=None, figsize=(14,12), title="VGG16 (transfer learning), Fashion-MNIST dataset")

# other KPIs
print('F1 score : {x:.4f}'.format(x=f1_score(y_test,y_pred,average='macro'))) 
print('Precision score : {x:.4f}'.format(x=precision_score(y_test,y_pred,average='macro')))
print('Recall score : {x:.4f}'.format(x=recall_score(y_test,y_pred,average='macro')))
