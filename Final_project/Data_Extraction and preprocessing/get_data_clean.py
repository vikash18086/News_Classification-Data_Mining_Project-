# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 23:52:01 2018

@author: Vikas
"""

import os
#import re
import stopword

data_folder = "../dataset/data_files"
folders = ["sport","tech"]

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
    
    if(i=="sport"):
        f4=open('sport.txt','a')
        f4.truncate(0)
        for j in x:
            f4.write(j)
    if(i=="tech"):
        f5=open('tech.txt','a')
        f5.truncate(0)
        for j in x:
            f5.write(j)
    del x[:]