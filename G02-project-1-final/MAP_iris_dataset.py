import scipy.stats
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.neighbors import KernelDensity
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
class NaiveBayes:
    def __init__(self, bandwidth):

        self.bandwidth = bandwidth
        self.X_train = np.array([])
        self.y_train = np.array([])

    def fit(self, X_train, y_train):
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)

        self.classes_ = np.sort(np.unique(self.y_train))

        training_sets = [self.X_train[self.y_train == yi] for yi in self.classes_]

        self.models_ = [KernelDensity(bandwidth=self.bandwidth).fit(Xi) for Xi in training_sets]

        self.logpriors_ = [np.log(Xi.shape[0] / self.X_train.shape[0]) for Xi in training_sets]

        return self

    def predict(self, X_test):   #return the predicted class for each element in X
        return self.classes_[np.argmax(self.predict_proba(X_test), 1)]

    def predict_proba(self, X_test): #calculate the logarithm of the likelihood probability
        logprobs = np.array([model.score_samples(X_test) for model in self.models_]).T

        result = np.exp(logprobs + self.logpriors_)

        return result / result.sum(1, keepdims=True)

data = pd.read_csv('iris.data', delim_whitespace=0)

bw = 0.337    # set the value of bandwidth
X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values
n_run = 100
averageAccuracy = 0
index = []
ac =[]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5)

bw = [0.1, 0.337, 1, 1.5, 2, 2.5, 3]
for i in bw:     #####

    start_time = time.time()
    nb = NaiveBayes(i)
    nb.fit(X_train, y_train)
    test_time = time.time()
    predictions = nb.predict(X_test)
    end_time = time.time()
    trainingTime = test_time - start_time
    testTime = end_time - test_time
    accuracy = 100 * accuracy_score(y_test, predictions)
    averageAccuracy += accuracy
    index.append(i)
    ac.append(accuracy)
    print(' Accuracy:' + str(accuracy) + '  \nTraining time:  ' + str(trainingTime) + ' s' + '\nTest time: ' + str(testTime) + '  s')

print(ac)
print(np.var(ac))
plt.plot(bw, ac)
plt.title("MAP classifier, Iris dataset")
plt.xlabel("Bandwidth")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.show()


averageAccuracy = averageAccuracy/n_run
cm = confusion_matrix(y_test, predictions)
print(cm)
cm_df = pd.DataFrame(cm,
                     index=['Iris_setora', 'Iris-versicolor', 'Iris_virginica'],
                     columns=['Iris_setora', 'Iris-versicolor', 'Iris_virginica'])

plt.figure(figsize=(7, 6))
sns.heatmap(cm_df, annot=True, fmt='g')
plt.title("MAP Classifier, Iris dataset \nAvg.accuracy: {:.2f}%, Training time: {:.3f} (ms), , Test time: {:.3f} (ms)".format(averageAccuracy, 1000*trainingTime, 1000*testTime))
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()