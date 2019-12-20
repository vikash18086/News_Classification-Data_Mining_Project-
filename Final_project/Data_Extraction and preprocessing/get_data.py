import os
import re
import stopword

data_folder = "../dataset/data_files"
folders = ["business_test" ,"entertainment_test"]

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
    return newlist

os.chdir(data_folder)

x = []
y = []

for i in folders:
    files = os.listdir(i)
    for text_file in files:
        file_path = i + "/" +text_file
        print ("reading file:", file_path)
        with open(file_path) as f:
            data = f.readlines()
        data = ' '.join(data)
        x.append(data+"\n************************************************\n")
        y.append(i)
    if(i=="business_test"):
        f1=open('business_test.txt','a')
        f1.truncate(0)              #remove old data from file
        for j in x:
            j = re.sub(r",",r"", j)
            j = re.sub(r"'", "", j)
            j = re.sub(r"\(",r"", j)
            j = re.sub(r"\)",r"", j)
            j = re.sub(r"\?",r"", j)
            j = re.sub(r"'s",r"", j)
            j = re.sub(r"[^A-Za-z0-9(),!?\'\`]",r" ", j)
            j = re.sub(r"[0-9]\w+|[0-9]",r"", j)
            #j = re.sub(r"\s{2,}",r" ", j)
            #j = remove_stopwords(j)
            j=j+"\n************************************************\n"
            
            f1.write(j)
            
        
        
    if(i=="entertainment_test"):
        f2=open('entertainment_test.txt','a')
        f2.truncate(0)
        for j in x:
            j = re.sub(r",",r"", j)
            j = re.sub(r"'", "", j)
            j = re.sub(r"\(",r"", j)
            j = re.sub(r"\)",r"", j)
            j = re.sub(r"\?",r"", j)
            j = re.sub(r"'s",r"", j)
            j = re.sub(r"[^A-Za-z0-9(),!?\'\`]",r" ", j)
            j = re.sub(r"[0-9]\w+|[0-9]",r"", j)
            #j = re.sub(r"\s{2,}",r" ", j)
            #j = remove_stopwords(j)
            j=j+"\n************************************************\n"
            f2.write(j)
    if(i=="politics_train"):
        f3=open('politics_train.txt','a')
        f3.truncate(0)
        for j in x:
            j = re.sub(r",",r"", j)
            j = re.sub(r"'", "", j)
            j = re.sub(r"\(",r"", j)
            j = re.sub(r"\)",r"", j)
            j = re.sub(r"\?",r"", j)
            j = re.sub(r"'s",r"", j)
            j = re.sub(r"[^A-Za-z0-9(),!?\'\`]",r" ", j)
            j = re.sub(r"[0-9]\w+|[0-9]",r"", j)
            #j = re.sub(r"\s{2,}",r" ", j)
            #j = remove_stopwords(j)
            j=j+"\n************************************************\n"
            f3.write(j)
    if(i=="sport_train"):
        f4=open('sport_train.txt','a')
        f4.truncate(0)
        for j in x:
           
            j = re.sub(r",",r"", j)
            j = re.sub(r"'", "", j)
            j = re.sub(r"\(",r"", j)
            j = re.sub(r"\)",r"", j)
            j = re.sub(r"\?",r"", j)
            j = re.sub(r"'s",r"", j)
            j = re.sub(r"[^A-Za-z0-9(),!?\'\`]",r" ", j)
            j = re.sub(r"[0-9]\w+|[0-9]",r"", j)
            #j = re.sub(r"\s{2,}",r" ", j)
            #j = remove_stopwords(j)
            j=j+"\n************************************************\n"
            f4.write(j)
    if(i=="tech_train"):
        f5=open('tech_train.txt','a')
        f5.truncate(0)
        for j in x:
            
            j = re.sub(r",",r"", j)
            j = re.sub(r"'", "", j)
            j = re.sub(r"\(",r"", j)
            j = re.sub(r"\)",r"", j)
            j = re.sub(r"\?",r"", j)
            j = re.sub(r"'s",r"", j)
            j = re.sub(r"[^A-Za-z0-9(),!?\'\`]",r" ", j)
            j = re.sub(r"[0-9]\w+|[0-9]",r"", j)
            #j = re.sub(r"\s{2,}",r" ", j)
            #j = remove_stopwords(j)
            j=j+"\n************************************************\n"
            f5.write(j)
    del x[:]
