from Supervised import *
from Unsupervised import *
from Cifar import *
import numpy as np
import matplotlib as plt
import pandas as pd
np.set_printoptions( precision = 3, suppress = True )
labels = ['Ankle boot', 'Bag', 'Sneaker', 'Shirt', 'Sandal', 'Coat', 'Dress', 'Pullover', 'Trouser', 'T-shirt/Top']
def cm_analysis(y_true, y_pred, labels, ymap=None, figsize=(6,6), title="Confusion matrix"):
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
    #plt.savefig("")
    plt.title(title)
    plt.show()
# General settings (you CAN change these):
cifar_use_threshold = False

# RBM hyperparameters (you CAN change these):
rbm_is_continuous	= True
rbm_visible_size	= 3072
rbm_hidden_size		= 50
rbm_batch_size		= 10
rbm_learn_rate		= 0.01
rbm_cd_steps		= 1
rbm_training_epochs = 20
rbm_report_freq		= 1
rbm_report_buffer	= rbm_training_epochs

# MLP hyperparameters (you CAN change these):
mlp_layer_sizes		= [ rbm_hidden_size, 20, 10 ]
mlp_batch_size		= 50
mlp_learn_rate		= 0.05
mlp_training_epochs = 20
mlp_report_freq		= 1
mlp_report_buffer	= mlp_training_epochs

# cifar training example counts
cifar_num_training_examples	  = 5000
cifar_num_validation_examples =	 0
cifar_num_testing_examples	  =	 1000

import numpy as np
from sklearn.metrics import confusion_matrix
cifar = Cifar( cifar_use_threshold )

training_digits,   training_labels	 = cifar.getTrainingData( cifar_num_training_examples )
validation_digits, validation_labels = cifar.getValidationData( cifar_num_validation_examples )
testing_digits,	   testing_labels	 = cifar.getTestingData( cifar_num_testing_examples )
rbm_name = 'rbm_' + str(rbm_visible_size) + '_' + str(rbm_hidden_size)
rbm = Rbm( rbm_name, rbm_visible_size, rbm_hidden_size, rbm_is_continuous )
rbm.train( training_digits, validation_digits, rbm_learn_rate, rbm_cd_steps, rbm_training_epochs, rbm_batch_size, rbm_report_freq, rbm_report_buffer )

# Encode datasets with RBM:
_, training_encodings = rbm.getHiddenSample( training_digits )
_, validation_encodings = rbm.getHiddenSample( validation_digits )
_, testing_encodings = rbm.getHiddenSample( testing_digits )

# Initialize and train MLP:
mlp_name = 'mlp_' + '_'.join( str(i) for i in mlp_layer_sizes )
mlp = Mlp( mlp_name, mlp_layer_sizes, 'sigmoid' )
mlp.train( training_encodings, training_labels, validation_encodings, validation_labels, mlp_learn_rate, mlp_training_epochs, mlp_batch_size, mlp_report_freq, mlp_report_buffer )

# Perform final test:
testing_guesses = mlp.predict( testing_encodings )
testing_error = mlp.getErrorRate( testing_labels, testing_guesses )
testing_accuracy = cifar_get_accuracy( testing_labels, testing_guesses )
print ('Final Testing Error Rate: %f' % ( testing_error ))
print ('Final Testing Accuracy: %f' % ( testing_accuracy ))