#import urllib
import requests as rq
import bs4
x=""
str1=''
str2=''
link=[]
#a=open('seed-hindi.html')
##l=[]
#l=data
categories=[]

temp=""
#s=''
#s=str(l).strip('[]')

def get_data(mylink):

    f1=open('DATA_3.txt','a')      
    print ('crawling : '+mylink);
    page_data=rq.get(mylink)
    soup=bs4.BeautifulSoup(page_data.text)
    
    alldata=soup.select('script')
    #print alldata
    
    for i in alldata:
        string=i.text
            
        newstr=string.encode('ascii','ignore')
        f1.write(str(newstr))
        
'''    
def dataget():
    for i in x:
        categories.append(i)
        temp=i.get_text()
        print (temp);
        str3=''
        for j in temp:
            if j.isalnum():
                str3=str3+j
            if j==',' or j=='.':
                str3=str3+j+'\n'                
            if j==' ':
                str3=str3+' '        
        b.write(str3)        
    b.close()
    a.close()
'''


def openlinks():
    for i in range(0,9):
        link.append('https://edition.cnn.com/2018/11/03/politics/trump-kavanaugh-montana-rally/index.html'+str(i))

#dataget()
#alllink()
openlinks()
for i in link:
    get_data(i)