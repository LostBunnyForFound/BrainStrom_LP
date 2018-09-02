# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 20:39:45 2018

@author: zsl
"""
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from gensim.models import KeyedVectors
nodeVectors = KeyedVectors.load_word2vec_format("data/LINE-master/windows/embedding_file_order1")
print("embedding loaded")
#i=0
#train_vector={}
#with open('data/LINE-master/windows/embedding_file_d50','r') as train_vector_file:
#    for line in train_vector_file:
#        if(i!=0):
#            items=line.split()
#            train_vector[int(items[0])]=[float(item) for item in items[1:]]
#        i=i+1
#print("embedding loaded")

DG=nx.read_edgelist("data/train.edgelist", nodetype=int,create_using=nx.DiGraph())
UDG=DG.to_undirected()
print("network loaded")

train_data=[]
with open('data/traindata_10W.txt','r') as train_file:
    for line in train_file:
        train_data.append([int(item) for item in line.split()])

test_data=[]
i=0
with open('data/test-public.txt','r') as test_file:
    for line in test_file:
        if(i!=0):
            items=line.split()
            test_data.append([int(item) for item in items[1:]])
        i=i+1

def calComNeigh(nodeA,nodeB,DG):
    if(DG.has_node(nodeA)and DG.has_node(nodeB)):
        com_pred=len(list(set(DG.predecessors(nodeA)) & set(DG.predecessors(nodeB))))
        com_suc=len(list(set(DG.successors(nodeA)) & set(DG.successors(nodeB))))
        return [com_pred,com_suc]
    else:
        return [0,0]
    
def findVector(num):
    if(num in nodeVectors):
        return 1
    else:
        return None

    
def calSim(nodeA,nodeB):
    vectorA=findVector(nodeA)
    vectorB=findVector(nodeB)
    if(vectorA is not None and vectorB is not None):
        return nodeVectors.similarity(nodeA,nodeB)
    else:
        return 0

def calAA(nodeA,nodeB):
    if(UDG.has_node(nodeA) and UDG.has_node(nodeB)):
        try:
            AA=nx.adamic_adar_index(UDG, [(nodeA, nodeB)])
            for u,v,p in AA:
                return p
        except ZeroDivisionError:
            return 0
    else:
        return 0
def calRA(nodeA,nodeB):
    if(UDG.has_node(nodeA) and UDG.has_node(nodeB)):
        try:
            AA=nx.resource_allocation_index(UDG, [(nodeA, nodeB)])
            for u,v,p in AA:
                return p
        except ZeroDivisionError:
            return 0
    else:
        return 0

train_similarity=[]
train_label=[]
for line in train_data:
    sim_line=calComNeigh(line[0],line[1],DG)
    sim_line.append(calSim(str(line[0]),str(line[1])))
    sim_line.append(calAA(line[0],line[1]))
    sim_line.append(calRA(line[0],line[1]))
    train_similarity.append(sim_line)
    train_label.append(line[2])

test_similarity=[]
test_raw=[]
for line in test_data:
    sim_line=calComNeigh(line[0],line[1],DG)
    sim_line.append(calSim(str(line[0]),str(line[1])))
    sim_line.append(calAA(line[0],line[1]))
    sim_line.append(calRA(line[0],line[1]))
    test_similarity.append(sim_line)

from sklearn.model_selection import train_test_split
train_sim, test_sim, train_lbl, test_lbl = train_test_split(
 train_similarity, train_label, test_size=1/7.0, random_state=0)

#import matplotlib.pyplot as plt
#
#plt.plot(train_similarity, train_label, 'rx')
#plt.ylabel("y ")
#plt.xlabel("X Similarity")
#plt.show()

from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier()
forest.fit(train_similarity,train_label)
pred_for_forest=forest.predict(test_similarity)

from sklearn.ensemble import AdaBoostClassifier
bdt = AdaBoostClassifier()
bdt.fit(train_similarity, train_label)
pred_for_bdt=bdt.predict(test_similarity)
#lr = LinearRegression().fit(train_sim, train_lbl)
#prediction=lr.predict(test_sim)

#clf = svm.SVC()
#clf.fit(train_sim, train_lbl)
#prediction=clf.predict(test_sim)
#prediction=[1 if item>=0.5 else 0 for item in prediction]

#i=0.0
#for item1,item2 in zip(prediction,test_lbl):
#    if item1>=0.5:
#        item1=1
#    else:
#        item1=0
#    if item1==item2:
#        i=i+1
#print(i/len(test_lbl))


#result=[]
#i=1
#for item in prediction:
#    result_line=[]
#    result_line.append(i)
#    result_line.append(item)
#    result.append(result_line)
#    i=i+1
#with open('data/logstics_10W_Emb_AA_RA.csv','w') as result_file:
#    result_file.write('Id,Prediction\n')
#    for line in result:
#        result_file.write('%d,%f\n'%(line[0],line[1]))


result_forest=[]
i=1
for item in pred_for_forest:
    result_line=[]
    result_line.append(i)
    result_line.append(item)
    result_forest.append(result_line)
    i=i+1
with open('data/forest_10W_Emb_AA_RA.csv','w') as result_file:
    result_file.write('Id,Prediction\n')
    for line in result_forest:
        result_file.write('%d,%f\n'%(line[0],line[1]))

result_bdt=[]
i=1
for item in pred_for_bdt:
    result_line=[]
    result_line.append(i)
    result_line.append(item)
    result_bdt.append(result_line)
    i=i+1
with open('data/bdt_10W_Emb_AA_RA.csv','w') as result_file:
    result_file.write('Id,Prediction\n')
    for line in result_bdt:
        result_file.write('%d,%f\n'%(line[0],line[1]))