# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 14:41:14 2018

@author: Vikas
"""

from bs4 import BeautifulSoup

soup = BeautifulSoup('<b class="boldest">Extremely bold</b>')
tag  = soup.b
type(tag)

#print(tag)
#print(tag.name)
#tag.name = "blockquotes"
#tag['another-attribute'] = 1
#tag['attribute-2'] = 2
#tag['id-1'] = 'verybold'
#print(tag)
#tag['id-2'] = 'very-very-bold'
#print(tag)
#del tag['id-1']
#del tag['attribute-2']
#print(tag)

### Multi valued attributes
#css-soup = BeautifulSoup('<p class = "body"></p>')
#css-soup.p['class']
####################change tag's string
#print(tag.string)
#tag.string.replace_with("Replace_string")
#print(tag.string)
