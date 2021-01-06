from __future__ import division
import math
from _csv import reader
import time
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def loadCsv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


def safeDiv(x,y):
    if y == 0:
        return 0
    return x/y


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

def strColToInt(dataset, column):
    class_values = [row[column] for row in dataset]
    unique = set(class_values)
    lookup = dict()
    for i, value in enumerate(unique):
        lookup[value] = i
    for row in dataset:
        row[column] = lookup[row[column]]
    return lookup

def strColToFloat(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def mean(numbers):
    return safeDiv(sum(numbers),float(len(numbers)))

def stdev(numbers):
    avg = mean(numbers)
    variance = float(sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1))
    return math.sqrt(variance)


def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries


def calculateProbability(x, mean, stdev):
    if stdev == 0:
        return 1
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
    trainingSet = loadCsv('optdigits.tra')
    strColToInt(trainingSet, len(trainingSet[0]) - 1)
    for i in range(0, len(trainingSet[0]) - 1):
        strColToFloat(trainingSet, i)
    testSet = loadCsv('optdigits.tes')
    strColToInt(testSet, len(testSet[0]) - 1)
    for i in range(0, len(testSet[0]) - 1):
        strColToFloat(testSet, i)
    print('Optdigits dataset includes train=%s and test=%s rows' % (len(trainingSet), len(testSet)))

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

    print('Accuracy:' + str(accuracy) + '  \nTraining time:  ' + str(trainingTime) +' s' + '\nTest time: ' + str(testTime) + '  s')
    cm = confusion_matrix(true_y, predictions)
    cm_df = pd.DataFrame(cm,
                         index=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                         columns=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_df, annot=True, fmt='g')
    plt.title("MLE Classifier, Digits dataset \nAccuracy: {:.2f}%, Training time: {:.3f} (ms), , Test time: {:.3f} (ms)".format(accuracy, 1000*trainingTime, 1000*testTime))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

main()