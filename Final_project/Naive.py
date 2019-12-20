# load the dataset
'''
data = open('dataset.csv').read()
labels=[]
texts= []
for i, line in enumerate(data.split("\n")):
    print line
    content = line.split()
    print content[0]
    labels.append(content[0])
    texts.append(content[1:])

# create a dataframe using texts and lables
traindataframe = pandas.DataFrame()
traindataframe['content']=content
traindataframe['label']=labels



import numpy as np
csv = np.genfromtxt ('dataset.csv', delimiter=",")
second = csv[:,0]
third = csv[:,1]
print second[0]
'''
import pandas
import csv
import numpy
from sklearn import model_selection , preprocessing , svm , metrics , naive_bayes, linear_model
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def training(classifier_used,training_feature,label,valid_feature):

    classifier_used.fit(training_feature,label)
    
    predictions=classifier_used.predict(valid_feature)
    #print(valid_y)
    r= metrics.accuracy_score(predictions,yvld)
    return r

content=[]
label=[]


readfile=csv.reader(open("dataset.csv", "rt"), delimiter=",")
x=list(readfile)
res=numpy.array(x).astype("str")
print (res.size)
print (type(res))
print (len(res))
for i in range (1,len(res)):
    content.append(res[i][0])
    label.append(res[i][1])
'''
print len(text)
print len(label)
print type(text)
#print text.size
print label[0]
print label[10]
print label[1500]
'''
train_dataframe = pandas.DataFrame()
train_dataframe['content']=content
train_dataframe['label']=label


model=model_selection.train_test_split(train_dataframe['content'], train_dataframe['label'],test_size=0.33)
xtrn=model[0]
xvld=model[1]
ytrn=model[2] 
yvld=model[3]
#print (xtrn)
#print ("\n")
#print ("\n")
#print (xvld)
#print ("\n")
#print ("\n")
#print (ytrn)
#print ("\n")
#print ("\n")
#print (yvld)
#print ("\n")
#print ("\n")
#print  (len(xtrn))
#print (len(xvld))
#print (len(ytrn))
#print (len(yvld))

e=preprocessing.LabelEncoder()
ytrn=e.fit_transform(ytrn)
yvld=e.fit_transform(yvld)

tfidf_vector = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}')
tfidf_vector.fit(train_dataframe['content'])
xtrn_tfidf=tfidf_vector.transform(xtrn)
xvalid_tfidf=tfidf_vector.transform(xvld)

#print (tfidf_vector)
#print (xtrn_tfidf)


tfidf_vector_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}',ngram_range=(2,3))
tfidf_vector_ngram.fit(train_dataframe['content'])
xtrn_tfidf_ngram =  tfidf_vector_ngram.transform(xtrn)
xvld_tfidf_ngram =  tfidf_vector_ngram.transform(xvld)


acc1=training(naive_bayes.MultinomialNB(),xtrn_tfidf,ytrn,xvalid_tfidf)#training Naive Bayes on tfidf features and get accuracy
print ("Naive Bayes+tfidf :",acc1)

#acc2=training(linear_model.LogisticRegression(),xtrn_tfidf,ytrn,xvalid_tfidf)#training on logistic regression with tfidf features
#print ("Logistic Regression+tfidf :",acc2)

acc3=training(svm.SVC(), xtrn_tfidf_ngram,ytrn,xvld_tfidf_ngram)#training SVM on ngram tfidf features and get accuracy
print ("svm+ngram tfidf :",acc3)


# SVM on Ngram Level TF IDF Vectors
accuracy = training(svm.LinearSVC(),xtrn_tfidf_ngram, ytrn, xvld_tfidf_ngram)
print ("SVM, N-Gram Vectors: ", accuracy)

#accuracy = training(svm.LinearSVC(), xtrn_count, ytrn, xvalid_count)
#print ("SVM, N-Gram Vectors: ", accuracy)


