import numpy as np
import pandas as pd
import os
import operator
from collections import Counter 
from sklearn.model_selection import train_test_split
import operator


base_path = 'data'

def read_data(base_path, filename):
	csv_file = os.path.join(base_path, filename)
	df = pd.read_csv(csv_file)
	X = df.iloc[:, :-1]
	y = df.iloc[:, -1]
	return X, y

def separate_class(X, y):
	separated = {}
	unique = np.unique(y)
	for i in range(len(unique)):
		X_by_class = X.loc[y == unique[i]]
		separated[unique[i]] = X_by_class
	return separated

def summarize(separated):
	summaries = {}
	for classValue, attributes in separated.items():
		for attribute in attributes:
			if classValue not in summaries:
				summaries[classValue] = []
			attribute_value = separated[classValue].loc[:, attribute]
			mean = np.mean(attribute_value)
			std = np.std(attribute_value)
			summaries[classValue].append((mean, std))
	return summaries

def calculate_class_probabilities(separate_class):
	class_probabilities = {}
	total_length = sum(len(values) for values in separate_class.values())
	for classValue in separate_class:
		numerator = np.shape(separate_class[classValue])[0]
		class_probabilities[classValue] = numerator / total_length
	return class_probabilities
	
def calculate_condition_probabilities(x, summaries):
	calculate_probabilities = {}
	for classValue in summaries:
		probabilities = []
		for i in range(len(summaries[classValue])):
			mean = summaries[classValue][i][0]
			std = summaries[classValue][i][1]
			exp = np.exp(-(np.power(x[i]-mean,2)/(2*np.power(std,2))))
			probability = exp/(np.power(2*np.pi*np.power(std, 2), 0.5))
			if classValue not in calculate_probabilities:
				calculate_probabilities[classValue] = []
			calculate_probabilities[classValue].append(probability)
		calculate_probabilities[classValue] = np.prod(np.array(calculate_probabilities[classValue]))  
	return calculate_probabilities

def predict(X_test):
	separated = separate_class(X, y)
	summaries = summarize(separated)
	class_probabilities = calculate_class_probabilities(separated)
	condition_probabilities = calculate_condition_probabilities(X_test, summaries)
	final_probabilities = {}
	for classValue in condition_probabilities:
		final_probabilities[classValue] = class_probabilities[classValue]*condition_probabilities[classValue]
	predict = max(final_probabilities.items(), key=operator.itemgetter(1))[0]
	return predict

X, y = read_data(base_path, '1.csv')

separated = separate_class(X, y)
calculate_class_probabilities(separated)
predict = predict([5,121,72,23,112,26.2,0.245,30])
print('Predict: {0}'.format(predict))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

y_pred = gnb.fit(X, y).predict([[2,88,58,26,16,28.4,0.766,22]])
print(y_pred)