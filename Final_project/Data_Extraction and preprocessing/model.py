import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import Word
import numpy as np
import nltk
#import stopword
nltk.download('wordnet')
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix

'''
def remove_stopwords(text):
    stop = stopword.allStopWords
    text=text.lower()
    text=text.split()
    newlist=[]
    stop_list=[]
    for i in stop:
        stop_list.append(i)
        
    for i in range (0,len(stop_list)):
        stop_list[i]=stop_list[i].lower()
    
    for i in text:
        if i not in stop_list: 
            newlist.append(i)
    s=" "
    newlist=s.join(newlist)
    
    newlist=newlist.replace('------------------','\n------------------\n')
    return newlist    
'''

def clean_str(string):
    string = re.sub(r"\'s",r"", string)
    string = re.sub(r"\'ve",r"", string)
    string = re.sub(r"n\'t",r"", string)
    string = re.sub(r"\'re",r"", string)
    string = re.sub(r"\'d",r"", string)
    string = re.sub(r"\'ll",r"", string)
    string = re.sub(r",",r"", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(",r"", string)
    string = re.sub(r"\)",r"", string)
    string = re.sub(r"\?",r"", string)
    string = re.sub(r"'",r"", string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]",r" ", string)
    string = re.sub(r"[0-9]\w+|[0-9]",r"", string)
    string = re.sub(r"\s{2,}",r" ", string)
    return string.strip().lower()

data = pd.read_csv('../dataset/dataset.csv')
x = data['news'].tolist()
y = data['type'].tolist()
dataset1=[]
for index,value in enumerate(x):
    print ("processing data:",index)
    x[index] = (y[index])
    x[index] = ' '.join([Word(word).lemmatize() for word in clean_str(value).split()])
    

dataset1=x
j=0;
f1=open('data_vikas.txt','a')
for i in range(len(dataset1)):
    f1.write(dataset1[i]+'\n-------------------------------------------------------\n')
    j=j+1;
'''
x="../model/data_vikas1.txt"
f2 = open(x)
doc_a = f2.read()
doc_a = remove_stopwords(doc_a)

f3=open('final_data1.txt','a')
f3.write(doc_a)
'''
vect = TfidfVectorizer(stop_words='english',min_df=2)
X = vect.fit_transform(x)
Y = np.array(y)
print("*****************************")
print(X)
print("*****************************")
print ("no of features extracted:",X.shape[1])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

print ("train size:", X_train.shape)
print ("test size:", X_test.shape)

model = RandomForestClassifier(n_estimators=300, max_depth=150,n_jobs=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
c_mat = confusion_matrix(y_test,y_pred)
kappa = cohen_kappa_score(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
print ("Confusion Matrix:\n", c_mat)
print ("\nKappa: ",kappa)
print ("\nAccuracy: ",acc)
print ("Total articles = ",j)
#print(dataset1)