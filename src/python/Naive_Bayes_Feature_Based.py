import numpy as np
import pandas as pd
import scipy as sp
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn import discriminant_analysis
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import ensemble
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score
import StringIO
import matplotlib
import matplotlib.pyplot as plt
import subprocess
import os
import random

data_filename = "../../data/labeled_data.txt"
max_sentence_length = 1000  # ignore
data = []
labels = []

with open(data_filename) as f:
    for line in f:
        s_len = len(line)
        cur = 0
        # split to segments of maximal length of max_sentence_length
        while (cur < s_len - 3):
            if (cur + max_sentence_length < s_len - 3):
                data.append(line[cur:cur+max_sentence_length].lower())
                labels.append(int(line[-2]))
                cur += max_sentence_length
            else:
                data.append(line[cur:-3].lower())
                labels.append(int(line[-2]))
                cur = s_len - 3

# Now, let's clean punctuations 
for i, sentence in enumerate(data):
    for char in ".,&%?'":
        data[i] = data[i].replace(char, ' ')
        
n = len(data)
x_train = []
y_train = []
x_test = []
y_test = []
for i in range(n):
    if random.randint(1, 8) <= 6:
        x_train.append(data[i])
        y_train.append(labels[i])
    else:
        x_test.append(data[i])
        y_test.append(labels[i])

vectorizer = CountVectorizer(encoding='latin1', stop_words=['and', 'or', 'before', 'a', 'an', 'am', 'the', 'at', 'by', 'br'], min_df=4)
x = vectorizer.fit_transform(x_train)
x = x.toarray()
word_counts = zip(x_train, x)
num_words = len(word_counts)
m = 12  # number of features
n = len(x)
vocab = vectorizer.vocabulary_.items()
vocab = sorted(vocab, key=lambda x: x[0])
def sentence_length(text):
    return len(text)

def average_word_length(words):
    return sum([len(word) for word in words]) / float(len(words))
    
def max_repeat(words):
    word_count = {}
    for word in words:
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1

    return max(word_count.values())
        
def number_of_repeated_words(words):
    word_count = {}
    for word in words:
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
    return sum([1 for word in word_count if word_count[word] > 1])

def contain_great(words):
    return int("great" in words)

def contain_people(words):
    return int("people" in words)

def contain_country(words):
    return int("country" in words)

def contain_we(words):
    return int("we" in words)

def contain_you(words):
    return int("you" in words)

def contain_them(words):
    return int("them" in words)

def contain_job(words):
    return int("job" in words)

def contain_I(words):
    return int("i" in words)

# calc_features
def calc_features(text):
    if text[-1] == '.':  # delete end punctuation
        text = text[:-1]
    words = text.split(' ')
    words = [word.lower() for word in words]
    return [sentence_length(text), average_word_length(words),\
           max_repeat(words), number_of_repeated_words(words),\
           contain_great(words), contain_people(words),\
           contain_country(words), contain_we(words), contain_you(words), contain_them(words),\
           contain_job(words), contain_I(words)]


data = []

def getFeaturesFromData():
    global n
    n = len(x)
    for i in range(n):
        data.append((y_train[i], calc_features(x_train[i])))
            
getFeaturesFromData()

import math
def getMean(arr):
    return sum(arr) / float(len(arr))

def getStdev(arr):
    avg = getMean(arr)
    variance = sum([pow(x-avg, 2) for x in arr])/float(len(arr)-1)
    return math.sqrt(variance)

means = []
stdevs = []
def processData():
    global means
    global stdevs
    means = [[], []]
    stdevs = [[], []]
    for label in range(2):
        for f in range(m):
            arr = [entry[1][f] for entry in data if entry[0] == label]
            means[label].append(getMean(arr))
            stdevs[label].append(getStdev(arr))
processData()

import sys
def mylog(x):
    if x == 0:
        return sys.float_info.min
    else:
        return math.log(x)

def calculateProbability(x, mean, stdev):
    if stdev < sys.float_info.epsilon:
        return sys.float_info.max
    else: 
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

def predictData(text):
    inputFeatures = calc_features(text)
    ans = [0] * 2
    total = [0] * 2
    for i in range(n):
        total[data[i][0]] += 1
    for label in range(2): 
       ans[label] += mylog(total[label] / (1.0 * total[0] + total[1]))
       for i in range(m):
           ans[label] += mylog(calculateProbability(inputFeatures[i], means[label][i], stdevs[label][i]))
    if ans[0] > ans[1]:
        return (0, "Donald Trump")
    else:
        return (1, "Hillary Clinton")


def testData(test_data):
    num_correct = 0
    num_total = len(test_data)
    for test_entry in test_data:
        num_correct += predictData(test_entry[1])[0] == test_entry[0]
    print "CORRECT ", num_correct, " out of ", num_total,\
            " percentage = %.2f" % (float(num_correct) / num_total)

test_data = zip(y_test, x_test)
testData(test_data)