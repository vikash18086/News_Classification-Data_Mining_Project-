# load the dataset
'''
data = open('dataset.csv').read()
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    print line
    content = line.split()
    print content[0]
    labels.append(content[0])
    texts.append(content[1:])

# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels



import numpy as np
csv = np.genfromtxt ('dataset.csv', delimiter=",")
second = csv[:,0]
third = csv[:,1]
print second[0]
'''

import random
import csv
import numpy as np
import pandas
from sklearn import model_selection , preprocessing , svm , metrics , naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
#from keras import layers, models, optimizers
label , text = [] ,  []


def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)

    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)

    if is_neural_net:
        predictions = predictions.argmax(axis=-1)

    return metrics.accuracy_score(predictions, valid_y)


reader = csv.reader(open("dataset.csv", "rb"), delimiter=",")



x = list(reader)
result = np.array(x).astype("string")
print (result.size)
print (type(result))
print (len(result))
#result = result.tolist()
print (type(result))
print (len(result))
np.random.shuffle(result)
print (type(result))
for i in range (1, len(result)):
    text.append(result[i][0])
    label.append(result[i][1])
'''
print len(text)
print len(label)
print type(text)
#print text.size
print label[0]
print label[10]
print label[1500]
'''

trainDF = pandas.DataFrame()
trainDF['text'] = text
trainDF['label'] = label


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

print train_x
print "\n"
print "\n"
print  valid_x
print "\n"
print "\n"
print train_y
print "\n"
print "\n"
print valid_y
print "\n"
print "\n"
print  len(train_x)
print len(valid_x)
print len(train_y)
print len(valid_y)

encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)

tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', max_features=5000)
tfidf_vect.fit(trainDF['text'])
xtrain_tfidf =  tfidf_vect.transform(train_x)
xvalid_tfidf =  tfidf_vect.transform(valid_x)

print tfidf_vect
print xtrain_tfidf

# ngram level tf-idf
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)

# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', ngram_range=(2,3), max_features=5000)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)

# create a count vectorizer object
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)




# SVM on Ngram Level TF IDF Vectors
accuracy = train_model(svm.LinearSVC(), xtrain_tfidf_ngram, train_y, xvalid_tfidf_ngram)
print "SVM, N-Gram Vectors: ", accuracy

accuracy = train_model(svm.LinearSVC(), xtrain_count, train_y, xvalid_count)
print "SVM, N-Gram Vectors: ", accuracy


accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf, train_y, xvalid_tfidf)
print "NB, WordLevel TF-IDF: ", accuracy

# Linear Classifier on Word Level TF IDF Vectors
accuracy = train_model(linear_model.LogisticRegression(), xtrain_tfidf, train_y, xvalid_tfidf)
print "LR, WordLevel TF-IDF: ", accuracy

# Naive Bayes on Character Level TF IDF Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_tfidf_ngram_chars, train_y, xvalid_tfidf_ngram_chars)
print "NB, CharLevel Vectors: ", accuracy

# Naive Bayes on Count Vectors
accuracy = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print "NB, Count Vectors: ", accuracy
