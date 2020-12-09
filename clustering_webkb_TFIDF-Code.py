# %%


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
import json
nltk.download('reuters')


# %%
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




# %%
import math
from scipy.spatial import distance
import statistics
from scipy.stats import pearsonr,entropy
def Manhattan(doc1,doc2):
    return distance.cityblock(doc1,doc2) 
def Euclidean(a, b):#distance
    return distance.euclidean(a,b)
def Cosine(a, b):#distance
    return distance.cosine(a,b)
def Jaccard(a, b):#distance
    return distance.jaccard(a,b)
def KL(a, b):
    return entropy(a,b)
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
def Smtp(a,b,variance):
    lemda=1.0
    simVal=0
    variance=1
    q1=1
    q2=1
    feature=np.zeros((total_features))
    sgn_g1=np.zeros((total_features))
    dval_g1=np.zeros((total_features))
    meanVal_g1=np.zeros((total_features))
    for k in range(0,len(feature)):
        if (a[k] > 0):
            sgn_g1[k]=sgn_g1[k]+1
            dval_g1[k]=dval_g1[k]+a[k]
        meanVal_g1[k] = dval_g1[k]
        if (b[k] > 0):
            sgn_g2[k]=sgn_g2[k]+1
            val_g2[k]=dval_g2[k]+b[k]
        meanVal_g2[k] = dval_g2[k]
    Ns_Sum=0
    Nu_Sum=0
    lemda=1.0
    term1=np.array((0.5*(1+np.exp(-1*(((meanVal_g2-meanVal_g1)/variance)**2))) *sgn_g2*sgn_g1 ) )  
    Ns_Sum=np.sum((term1+((lemda)*(q2-sgn_g2)*sgn_g1)+((lemda)*sgn_g2*(q1-sgn_g1))))
    Nu_Sum=np.sum((sgn_g2*sgn_g1)+((q2-sgn_g2)*sgn_g1)+(sgn_g2*(q1-sgn_g1)))
    F_G1G2=(Ns_Sum / Nu_Sum)  
    simVal=((F_G1G2+lemda)/(1+lemda))
    return (1-simVal)
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
    return 1-(0.5*(ECSM_P1(doc1,doc2)+ECSM_P2(doc1,doc2)))
# %%
def most_common(lst):
    return max(set(lst), key=lst.count)

def tokenize1(documents):
    tokens=[]
    content= documents
    curr_doc_index=index;
    tokens=(word_tokenize(content))
    # lemmatizing documnets

    # removing stopwords
    tokens= [token.lower() for token in tokens ]
    tokens = [token for token in tokens if token not in stopwords]
    tokens= [token for token in tokens if token.isalpha()]
   # tokens= [token for token in tokens if len(token)>3 ]
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
  #  tokens= [token.lower() for token in tokens ]
  #  tokens = [token for token in tokens if token not in stopwords]
   # tokens= [token for token in tokens if token.isalpha()]
    tokens= [token for token in tokens if len(token)>3 ]
    return tokens
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
def sorted_tfs(tfs,n):
    doc,terms=tfs.shape
    ind=(np.argsort(-(np.asarray(tfs.sum(axis=0)).ravel())))
    scores=np.zeros((doc,n))
    for i in range(0,len(documents)):
        print(i)
        for j in range(0,n):
            if(tfs[i,ind[j]]!=0):
                scores[i,j]=tfs[i,ind[j]]
    return scores


def initialize_clusters(points, k):
    print("""Initializes clusters as k randomly selected points from points.""")
    print(points.shape[0],k)
    return points[np.random.randint(points.shape[0], size=k)]
    
# Function for calculating the distance between centroids
def get_distances(centroid, points,metric):
    dist=[]
    func = globals()[metric]
    variance=[]
    
    if(metric=="smtp"):
        row,col=points.shape
        print("row",row)
        print("col",col)
        for c in range(0,col):
            variance.append(np.std(points[:,c]))
        for pnt in points:
            if (not np.any(pnt)):
                dist.append(func(pnt,centroid,variance)) 
            else:
                 dist.append(-1)
    else:
        for pnt in points:
            if (np.any(pnt)):
                dist.append(func(pnt,centroid)) 
            else:
                dist.append(-1) 
        """Returns the distance the centroid is from each data point in points."""
    maxVal=max(dist)
    dist=[maxVal if x==-1 else x for x in dist]          
    return dist
def kMeanClustering(X,k,metric):
    k = k
    X=arr               
    maxiter = 50
    # Initialize our centroids by picking random data points
    print(X.shape)
    centroids = initialize_clusters(X, k)
    print("centroid shape")
    print (centroids.shape)

    # Initialize the vectors in which we will store the
    # assigned classes of each data point and the
    # calculated distances from each centroid
    classes = np.zeros(X.shape[0], dtype=np.float64)
    distances = np.zeros([X.shape[0], k], dtype=np.float64)

    # Loop for the maximum number of iterations
    for i in range(maxiter):

        # Assign all points to the nearest centroid
        for i, c in enumerate(centroids):
            
            distances[:, i] = get_distances(c, X,metric=metric)

        # Determine class membership of each point
        # by picking the closest centroid
        classes = np.argmin(distances, axis=1)

        # Update centroid location using the newly
        # assigned data point classes
        for c in range(k):
            centroids[c] = np.mean(X[classes == c], 0)

    return classes

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 
def accuracy_score(y_true, y_pred):
    clusters=set(y_pred)
    mostCommon=0
    totalDocs=len(y_true)
    for y in clusters:
        indices = [i for i, x in enumerate(y_pred) if x == y]
        True_Values=[y_true[j] for j in indices]
        common=(Counter(True_Values).most_common(1))
        #print(common)
        mostCommon+=(common[0])[1]
    accuracy_score=mostCommon/totalDocs
    return accuracy_score
        
def entropy_score(y_true, y_pred):
    clusters=set(y_pred)
    labels=set(y_true)
    n=len(y_true)
    p=len(labels)
    en_sum=0
    for y in clusters:
        indices = [i for i, x in enumerate(y_pred) if x == y]
        True_Values=[y_true[j] for j in indices]
        ni=len(True_Values)
        sum_nij=0
        for j in labels:
            nij=True_Values.count(j)
            if(nij>0):
                sum_nij+=(-1*(nij/ni)*np.log(nij/ni))
        en_sum+=ni*sum_nij   
    entropy_score=en_sum/(np.log(p)*n)
    return entropy_score


def PrintDetails(metric="smtp",time=None,Arr=None):
    fname='kMean_clustering_WebKb_TFIDF'+str(metric)+'_'+str(k)+'.txt'
    tables = open(fname, 'w')  
    x = PrettyTable()
    x.field_names = ["Measure","Value"]
    print("metric",metric)
    print("time",time)
    tables.write("metric\t"+str(metric)+"\n")
    tables.write("time\t"+str(time)+"\n")
    average=0
    for i in range(0,len(Arr)):
        x.add_row([(Arr[i])[0],(Arr[i])[1]])   
    tables.write(str(x))
    print (x) 
    tables.close()
def Clustering(arr=None,k=1,metric='Euclidean'):
    
    time1= time.time()
    classes=kMeanClustering(X=arr,k=k,metric='Euclidean')
    time2= time.time()
    y_pred=classes#[categories[ind] for ind in classes]
    purity=("purity",purity_score(labels, y_pred))
    accuracy=("Accuracy",accuracy_score(labels, y_pred))
    entropy=("Entropy",entropy_score(labels, y_pred))
    homogeneity=("homogeneity",homogeneity_score(labels, y_pred))
    completeness=("completeness",completeness_score(labels, y_pred))
    valArr=[]
    valArr.append(accuracy)
    valArr.append(entropy)
    valArr.append(purity)
    valArr.append(homogeneity)
    valArr.append(completeness)
    #purity
    mArr=["Purity","Homogeneity","Completeness"]

    PrintDetails(metric=metric,time=str(timedelta(seconds=(time2-time1))),Arr=valArr)
   
    

# %%
count=0
documents=[]
labels=[]
rawData=[]
token_dict = dict()
path = "webkb-data.gtar\\webkb"
loadWebKbData(path=path,documents=documents,labels=labels)
#loadReutersData(documents=documents,labels=labels)
print(len(documents))
categories=list(set(labels))
totalDocs= len(documents)
print (totalDocs)
stopwords = stopwords.words('english')
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
vocabulary = set()
term_docs=[]
terms=[]
totalDocs= len(documents)
print (len(documents))
for index in range(0, len(documents)):
    content= documents[index]
    curr_doc_index=index;
    tokens=tokenize1(content)
    tf=collections.Counter(tokens)
    #print tf
   # print index
    term_docs.append(tf)
    terms.append(tokens)
#for index in range(0,len(documents)):
   # print("index:",index)
   # print(term_docs[index])

print ("vectorize")
tfidf = TfidfVectorizer(tokenizer=tokenize1)
print ("fit transform")
tfs = tfidf.fit_transform(documents)
   
sorted_scores=display_scores(tfidf, tfs)
k=0
n=len(sorted_scores)
groups=groupData(labels,categories)   
index=0
print (totalDocs)
print (len(term_docs))
arr=sorted_tfs(tfs,n)
kList=[5,10,16,32,len(categories)]
for k in kList:
    print("###############")
    measures=["Manhattan","Euclidean","Cosine","KL","smtp","extendedJaccard","Jaccard","EBLAB_SM"]
    for met in measures:
        print("metric", met)
        print("k",k)
        Clustering(arr=arr,k=k,metric=met)


# %%


# %%


# %%


# %%


# %%


# %%
