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
import argparse
import sys
import math

def argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', default="../../data/labeled_data.txt")
    parser.add_argument('--eval_file', default="")
    parser.add_argument('--max_length', default=500, type=int)
    args = parser.parse_args()
    return args

def train(args):
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
	global x_train, y_train, x_test, y_test
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

	word_to_index = {}
	n = len(x_train)
	num = 0
	for i in range(n):
    	words = x_train[i].split()
    	for word in words:
        	if not word_to_index.has_key(word):
            	word_to_index[word] = num
            	num += 1

	total = [0] * 2
	for i in range(n):
    	total[y_train[i]] += 1
    
	counts = []
	for label in range(2):
    	counts.append([0] * num)
    	for i in range(n):
        	if y_train[i] == label:
            	words = x_train[i].split()
            	for word in words:
                	counts[label][word_to_index[word]] += 1

	# helper functions to get stats
	def get_word_stats():
    	for key in word_to_index:
        	c0 = counts[0][word_to_index[key]]
        	c1 = counts[1][word_to_index[key]]
        	if abs(c0 - c1) < (c0 + c1) / 4:
            	continue
        	if counts[0][word_to_index[key]] + counts[1][word_to_index[key]] > 50:
            	print "key=", key, " counts=", counts[0][word_to_index[key]], counts[1][word_to_index[key]]

	# p[label][word] = log(p(word | label))
	def mylog(x):
    	if x == 0:
        	return sys.float_info.min
    	else:
        	return math.log(x)

	def getProb(cnt):
    	tot = sum(cnt)
    	return [float(x) / tot for x in cnt]
	def calculateTable(alpha = 1):
    	global p
    	p = []
    	for label in range(2):
        	p.append([0] * num)
        	arr = getProb([x + alpha for x in counts[label]])
        	p[label] = [mylog(x) for x in arr]
    calculateTable()


args = argument()
data_filename = args.data_file
eval_filename = ""
max_sentence_length = args.max_length
data = []
labels = []

def predictLabel(text):
    words = text.split()
    ans = [0] * 2
    for label in range(2):
        ans[label] = mylog(float(total[label]) / sum(total))
        for word in words:
            if not word_to_index.has_key(word):
                continue  # ignore for now
            else:
                ans[label] += p[label][word_to_index[word]]
    if ans[0] > ans[1]:
        return 0
    else:
        return 1

def testData(test_data):
    num_total = len(test_data)
    num_correct = 0
    for test_entry in test_data:
        num_correct += (predictLabel(test_entry[1]) == test_entry[0])
    return "%d out of %d: correct %.2lf" % (num_correct, num_total, float(num_correct) / num_total)

test_data = zip(y_test, x_test)
testData(test_data)


# Evaluation
label_names = {0:'Trump', 1:'Clinton'}
print 'eval_file = ', eval_filename
