#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# %%
# importing all the required modules

from importlib import reload
import nltk
from bs4 import BeautifulSoup
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import errno
import string
from nltk.corpus import reuters
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.text import TextCollection
import collections
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from prettytable import PrettyTable
import time
from datetime import timedelta
from sklearn.metrics import recall_score,precision_score,average_precision_score,f1_score,accuracy_score
from sklearn.preprocessing import label_binarize
from imblearn.metrics import geometric_mean_score
from sklearn import metrics
from sklearn.metrics.cluster import homogeneity_score,completeness_score
import statistics
import math
import sklearn.metrics 
from sklearn.model_selection import KFold
import csv
from sklearn.model_selection import train_test_split
from collections import Counter
import math
from scipy.spatial import distance
import statistics
from scipy.stats import pearsonr,entropy
from nltk.stem import WordNetLemmatizer
nltk.download('reuters')
def loadReutersData(documents,labels):
    categories_list=['acq','crude','earn','grain','interest','money-fx','ship','trade']
    docCount=0
    for i in range(0,len(categories_list)):
        category_docs = reuters.fileids(categories_list[i])
        print (categories_list[i])
        for document_id in reuters.fileids(categories_list[i]):
            if(len(reuters.categories(document_id))==1):
                content=str(reuters.raw(document_id))
                soup = BeautifulSoup(content)
                content=soup.get_text()
                documents.append(content)
                docCount+=1
                labels.append(str(reuters.categories(document_id)))
def loadWebKbData(path, documents,labels):
    print(path)
    for root, dirs, files in os.walk(path):  
        for filename in files:
            try:
                #print root
                name = os.path.join(root, filename)
                #print name
                end=len(name)-len(filename)
                test=name[len(path)+1:end]
                for i in range(0,len(test)):
                    if test[i]=='\\':
                        labels.append(test[0:i])
                        break
                f = open(name, "rb").read()
                f=f.decode('ISO-8859-1', 'ignore')
                content=str(f)
                rawData.append(f)
                soup = BeautifulSoup(content)
                content=soup.get_text()
                documents.append(content)  
            except IOError as exc:
                if exc.errno != errno.EISDIR:
                       raise
'''
*****************************************Distnace measures*******************************************

'''
def Manhattan(doc1,doc2):
    return distance.cityblock(doc1,doc2) 
def Euclidean(a, b):#distance
    return distance.euclidean(a,b)
def Cosine(a, b):#distance
    return distance.cosine(a,b)

def KL(a, b):
    return entropy(a,b)
def Jaccard(a, b):#distance
    return distance.jaccard(a,b)

def extendedJaccard(a,b):
    vector1=[0 if x==0 else 1 for x in a]
    vector2=[0 if x==0 else 1 for x in b]
    dot=np.dot(vector1,vector2)
    sum1=np.sum(vector1)
    sum2=np.sum(vector2)
    denom=math.sqrt(sum1)+math.sqrt(sum2)-dot
    if(denom!=0):
        return 1.0 - (float(dot)/(denom))
    else:
        return -1
def smtp(doc1,doc2,var):
    lemda=1
    a=set(np.nonzero(doc1)[0])  #indices of non zero elements in doc1
    b=set(np.nonzero(doc2)[0])  #indices of non zero elements in doc2
    intersection=a.intersection(b) # indices where both docs have non zero elements
    union=a.union(b) # indices where either doc has non zero elements
    d1=np.array(list(a-intersection)) # doc1 !=0 and doc2=0
    d2=np.array(list(b-intersection)) # doc1 =0 and doc2!=0
    doc1=np.array(doc1)
    doc2=np.array(doc2)
    Nstar=0
    intersection=np.array(list(intersection))
    if (len(intersection)>0):
        term1=np.exp(-1*np.square(( doc1[intersection]-doc2[intersection] )/var[intersection]))
        Nstar=sum(0.5* (1+term1)) +lemda* -1 *(len(d1)+len(d2))
    else:
        Nstar=lemda* -1 *(len(d1)+len(d2))   
    Nunion=len(intersection)+len(d1)+len(d2)
    smtp=((Nstar/Nunion)+lemda)/(1+lemda)
    return smtp
def ECSM_P1(doc1,doc2):
    simValue=0
    N=len(doc1)#total_features;
    a=set(list(np.nonzero(doc1 != 0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2 != 0)[0]))  #indices of non zero elements in doc2NSMT
    Nab=len(a.intersection(b))
    F=len(a.union(b))-Nab
    simValue=(1-F/N)
    return simValue
def ECSM_P2(doc1,doc2):
    a=set(list(np.nonzero(doc1 != 0)[0]))  #indices of non zero elements in doc1
    b=set(list(np.nonzero(doc2 != 0)[0]))  #indices of non zero elements in doc2NSMT
    sim=0
    if (len(a)+len(b))!=0:
        sim=(2*len(a.intersection(b)))/(len(a)+len(b))
    return sim

def EBLAB_SM(doc1,doc2):
    return 0.5*(ECSM_P1(doc1,doc2)+ECSM_P2(doc1,doc2))
'''
***************************************** End of Distnace measures*******************************************

'''

def most_common(lst):
    return max(set(lst), key=lst.count)
def tokenize1(documents):
    tokens=[]
    content= documents
    tokens=(word_tokenize(content))
    tokens= [token.lower() for token in tokens ]
    tokens = [token for token in tokens if token not in stopwords]
    tokens= [token for token in tokens if token.isalpha()]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens= [token for token in tokens if len(token)>3 ]
    return tokens

def display_scores(vectorizer, tfidf_result):
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
def sorted_tfs(tfs,n):
    doc,terms=tfs.shape
    ind=(np.argsort(-(np.asarray(tfs.sum(axis=0)).ravel())))
    scores=np.zeros((doc,n))
    for i in range(0,len(documents)):
        for j in range(0,n):
            if(tfs[i,ind[j]]!=0):
                scores[i,j]=tfs[i,ind[j]]
    return scores
def count_values_in_range(series, range_min, range_max):

    # "between" returns a boolean Series equivalent to left <= series <= right.
    # NA values will be treated as False.
    return series.between(left=range_min, right=range_max).sum()

class CustomKNN:

    #constructor

    def __init__(self,k = 1,metric="euclidean",termOccurance=None, docOccurance=None):

        
        self.k=k
        #print ("KNN",self.k)
        self.metric=metric
        self.trainingData=[]
        self.trainLabels=[]
        self.smtp=None
        self.termOccurance=termOccurance
        self.docOccurance=docOccurance
         
    
         

    def fit(self, training_data, trainLabels ):
        self.trainingData=training_data
        self.trainLabels=trainLabels
        
            
       
    def predict(self, testData):
        train=self.trainingData
        var=np.zeros(train.shape[1])
        if(self.metric=="smtp" ):
            var=np.var(train,axis=0)
            print("Vrainace is",var)
        distList=["KL","extendedJaccard","Euclidean","Cosine","Jaccard","Manhattan","Jaccard"]#distance
        func=globals()[self.metric]
        predLabel=[]
        for kk in range(0,60):
            temp=[]
            predLabel.append(temp)
        for i in(range(0,len(testData))):
            
            print(i)
            dist=[]
            for j in(range(0,len(train))):
                if ((not np.any(testData[i])) or (not np.any( train[j]))):
                    if(self.metric in distList):
                        dist.append(-1)
                    else:
                        dist.append(0)
                else:
                    if(self.metric=="smtp"):
                        #var=np.var(train,axis=1)
                        dist.append(smtp(testData[i],train[j],var))
                    else:
            
                        dist.append(func(testData[i], train[j]))     
            flag=-1
            if(self.metric in distList):
                maxVal=max(dist)
                dist=[maxVal if x==-1 else x for x in dist]
                flag=1
            for kk in range(0,60):
                nn=(kk*2)+1
                if(flag==-1):
                    neigh= np.argpartition(np.array(dist), len(dist) - nn)[-nn:]
                else:
                    neigh= (np.argpartition(np.array(dist),nn))[:nn]
                neighLabels=[self.trainLabels[ind] for ind in neigh]
                label_predict=(most_common(neighLabels))
                predLabel[kk].append(label_predict)


        return predLabel
        
def dislay_tfidf(vectorizer,tfidf_result):
    print(vectorizer.get_feature_names())
    print(tfidf_result)
    
def display_scores(vectorizer, tfidf_result):
    scores = zip(vectorizer.get_feature_names(),
                 np.asarray(tfidf_result.sum(axis=0)).ravel())
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return sorted_scores
def groupData(labels,categories):

    print (categories)
    groups=[]
    for i in range(0,len(categories)):
        groups.append([0,0,0])
    totalDocs=len(labels)
    print (totalDocs)
    for i in range(0,totalDocs):
        tmp=(categories.index(labels[i]))
        groups[tmp].append(i)
    for i in range(0,len(categories)):
        del (groups[i])[0:3]
    return groups
def PrintDetails(metric="smtp",time=None,measure= "Accuracy",Arr=None):
    x = PrettyTable()
    x.field_names = ["k","DataSet1"]
    print("metric",metric)
    print("time",time)
    print("measure",measure)
    tables.write("metric\t"+str(metric)+"\n")
    tables.write("time\t"+str(time)+"\n")
    tables.write("measure\t"+str(measure)+"\n")
    average=0
    for i in range(0,len(Arr)):
        x.add_row([(i*2)+1,Arr[i]]) 
        average+=Arr[i]
    x.add_row(["Average",average/len(Arr)])   
    tables.write(str(x))
    print (x)
    
def classify(k=3,metric="euclidean",termOccurance=None, docOccurance=None):
    time1= time.time()
    accuracy=[]
    precision=[]
    Recall=[]
    fMeasure=[]
    gMeasure=[]
    averageMeanPrecison=[]
    classifier =CustomKNN(k=k,metric=metric,termOccurance=termOccurance,docOccurance=docOccurance)  
    classifier.fit(train_data, train_labels)
    y_pred = classifier.predict(test_data) 
    time2= time.time()

    for i in range(0,len(y_pred)):
        #Accuracy
        #print(len(test_labels))
       # print(len(y_pred[i]))
    
        accuracy.append(accuracy_score(test_labels, y_pred[i]))
        
        #PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="Accuracy",Arr=accuracy)
        #Precison
        precision.append(precision_score(test_labels, y_pred[i],average='macro'))
         
        #PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="Precision",Arr=precision)
        #Recall
        Recall.append(recall_score(test_labels, y_pred[i],average='macro'))
           
        #PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="Recall",Arr=Recall)
        #f measure
        fMeasure.append(f1_score(test_labels, y_pred[i],average='macro'))
       # PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="F Measure",Arr=fMeasure)
        #g measure
        gMeasure.append(geometric_mean_score(test_labels, y_pred[i],average='macro'))
        #PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="g Measure",Arr=gMeasure)
        #Average Mean Precision
        test = label_binarize(test_labels, classes=categories)
        pred = label_binarize( y_pred[i], classes=categories)
        averageMeanPrecison.append(average_precision_score(test, pred))
        #PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="Average Mean Precison",Arr=averageMeanPrecison)

    #Accuracy
    PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="Accuracy",Arr=accuracy)
    #Precison
    PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="Precision",Arr=precision)
    #Recall
    PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="Recall",Arr=Recall)
    #f measure
    PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="F Measure",Arr=fMeasure)
    #g measure
    PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="g Measure",Arr=gMeasure)
    #Average Mean Precision
    PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),measure="Average Mean Precison",Arr=averageMeanPrecison)
dataset='reuters'
count=0
documents=[]
labels=[]
rawData=[]
token_dict = dict()
if dataset=='webkb':
    print("processing webkb data")
    path = "webkb-data.gtar\\webkb"
    loadWebKbData(path=path,documents=documents,labels=labels)
else:
    print("processing reuters data")
    loadReutersData(documents=documents,labels=labels)
print(len(documents))
categories=list(set(labels))
totalDocs= len(documents)
print (totalDocs)
stopwords = stopwords.words('english')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
vocabulary = set()
totalDocs= len(documents)
print (len(documents))
vocabulary = set()
print ("vectorize")
tfidf = TfidfVectorizer(tokenizer=tokenize1)
print ("fit transform")
tfs = tfidf.fit_transform(documents)
sorted_scores=display_scores(tfidf, tfs)
#dislay_tfidf(tfidf, tfs)
feat=[10,50,100,200,350,3000,6000,len(sorted_scores)]
for n in feat:
    groups=groupData(labels,categories) 
    arr=sorted_tfs(tfs,n)
    print (arr.shape)
    print ("start")
    xTrain, xTest, yTrain, yTest = train_test_split(arr, labels, test_size = 0.3, random_state = 42,stratify=labels)
    train_data=xTrain
    test_data=xTest
    train_labels=yTrain
    test_labels=yTest
    k=1
    measures=["Manhattan","Euclidean","Cosine","KL","smtp","extendedJaccard","Jaccard","EBLAB_SM"]
    for met in measures:
        k=1
        print("###############")
        fname='table_'+dataset+'_'+met+'_Tfidf_'+str(n)+'.txt'
        tables = open(fname, 'w')
        print ("nTerms",n)
        tables.write("No. of features\t"+str(n)+"\n")
        classify(k=k,metric=met)
        tables.close()
        print("###############")






# %%


# %%


# %%


# %%

