from __future__ import division
import csv
from csv import reader
import random
import math
import time
from sklearn.metrics import confusion_matrix, accuracy_score
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def safe_div(x,y):
    if y==0:
        return 0
    return x/y

def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    trainSet = []
    copy = list(dataset)
    while len(trainSet) < trainSize:
        index = random.randrange(len(copy))
        trainSet.append(copy.pop(index))
    return [trainSet, copy]


def str_column_to_int(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def mean(numbers):
    return safe_div(sum(numbers),float(len(numbers)))


def stdev(numbers):
    avg = mean(numbers)
    variance = float(sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1))
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    print(summaries)
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    print(separated)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    exponent = math.exp(-(math.pow(x - mean, 2) / float(2 * math.pow(stdev, 2))))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent


def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] *= calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions


def main():
    filename = 'iris.data'
    n_run = 100
    splitRatio = 0.5
    averageAccuracy = 0
    ac = []
    index = []
    for i in range(0, n_run):
        index.append(i)
        dataset = load_csv(filename)
        str_column_to_int(dataset, len(dataset[0]) - 1)
        for i in range(0, len(dataset[0]) - 1):
            str_column_to_float(dataset, i)
        trainingSet, testSet = splitDataset(dataset, splitRatio)
        print('Split iris dataset: %s rows into train=%s and test=%s rows' % (len(dataset), len(trainingSet), len(testSet)))

        start_time = time.time()

        summaries = summarizeByClass(trainingSet)
        test_time = time.time()
    # test model
        predictions = getPredictions(summaries, testSet)

        end_time = time.time()
        trainingTime = test_time - start_time
        testTime = end_time - test_time

        true_y = []
        for element in testSet:
            true_y.append(element[-1])
        accuracy = 100 * accuracy_score(true_y, predictions)
        averageAccuracy += accuracy
        ac.append(accuracy)

        print('Accuracy:' + str(accuracy) + '  \nTraining time:  ' + str(trainingTime) + ' s' + '\nTest time: ' + str(testTime) + '  s')

    print(ac)
    variance = np.var(ac)
    print(variance)
    plt.plot(index, ac)
    plt.title("MLE classifier, Iris dataset")
    plt.xlabel("Index of run")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.show()


    averageAccuracy = averageAccuracy/n_run
    cm = confusion_matrix(true_y, predictions)
    cm_df = pd.DataFrame(cm,
                         index=['Iris_setora', 'Iris-versicolor', 'Iris_virginica'],
                         columns=['Iris_setora', 'Iris-versicolor', 'Iris_virginica'])

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_df, annot=True)
    plt.title(
            "MLE Classifier, Iris dataset \nAvg.accuracy: {:.2f}%, Training time: {:.3f} (ms), , Test time: {:.3f} (ms)".format(
                averageAccuracy, 1000*trainingTime, 1000*testTime))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

main()