# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 09:19:16 2021

@author: 33646
"""

import monkdata as m
import dtree as dt

import numpy as np
import matplotlib.pyplot as plt

dataset1 = m.monk1
entropy1 = dt.entropy(dataset1)
# =============================================================================
# dataset2 = m.monk2
# entropy2 = dt.entropy(dataset2)
# dataset3 = m.monk3
# entropy3 = dt.entropy(dataset3)
# =============================================================================
print("Entropy computing")
print("dataset 1: ", entropy1)
# =============================================================================
# print("dataset 2: ", entropy2)
# print("dataset 3: ", entropy3)
# =============================================================================

print("\nInformation gain dataset1")
for i in range(6):
    print("Attribute "+str(i)+": ", dt.averageGain(dataset1, m.attributes[i]))
# =============================================================================
# print("\nInformation gain dataset2")
# for i in range(6):
#     print("Attribute "+str(i)+": ", dt.averageGain(dataset2, m.attributes[i]))
# print("\nInformation gain dataset3")
# for i in range(6):
#     print("Attribute "+str(i)+": ", dt.averageGain(dataset3, m.attributes[i]))
# 
# =============================================================================

# =============================================================================
# gainMaxAtt_ind = 4
# for value in range(len(m.attributes[gainMaxAtt_ind].values)): #several values of attribute 4: several subset
#     for att_ind in range(6): #several values 
#         if att_ind != gainMaxAtt_ind:
#             print("A"+str(gainMaxAtt_ind)+"="+str(value)+", A"+str(att_ind))
#             print("\n",dt.averageGain(subset_monk1[value], m.attributes[att_ind]))
# =============================================================================

root = 4
subsets = []
dataset1 = m.monk1
for val in range(1,5):
        print("A5= ", val)
        subset = dt.select(dataset1, m.attributes[root], val)
        for i in range(6):
                print("Attribute "+str(i+1)+": ", dt.averageGain(subset, m.attributes[i]))


    
dt.averageGain(dataset1, m.attributes[i])
          
NB_TRAINDATASETS = 3
NB_ATTRIBUTES = 6
## Assignement 1


# =============================================================================
# datasets = [m.monk1, m.monk2, m.monk3]
# entropy = [dt.entropy(datasets[k]) for k in range(3)]
# =============================================================================

## Assignement 3



# =============================================================================
# root_info_gain = np.zeros((NB_TRAINDATASETS, NB_ATTRIBUTES))
# max_att_idx = []
# for ds_idx, ds in enumerate(datasets):
#     for att_idx in range(NB_ATTRIBUTES):
#         root_info_gain[ds_idx, att_idx] = dt.averageGain(ds, m.attributes[att_idx])
#     max_att_idx.append(np.where(root_info_gain[ds_idx,:] == np.amax(root_info_gain[ds_idx,:]))[0][0])
#         
# # ## 5. Building Decision Trees
# 
# subsets = [[], [], []]
# for ds_idx, ds in enumerate(datasets):
#     for val in range(1, len(m.attributes[max_att_idx[ds_idx]].values)+1):
#         for att_idx in range(NB_ATTRIBUTES):
#             subsets[ds_idx].append(dt.select(ds, m.attributes[max_att_idx[ds_idx]], val))
# 
# firstfloor_gain = [[], [], []]
# for ds_idx in range(NB_TRAINDATASETS):
#     root = max_att_idx[ds_idx]
#     for val in range(1, len(m.attributes[root].values)+1): #several values -> several subset
#         for att_idx in range(NB_ATTRIBUTES): #several attributes
#             if att_idx != root: #except the root
#                 info_gain = dt.averageGain(subsets[ds_idx][val], m.attributes[att_idx])
#                 firstfloor_gain[ds_idx].append((info_gain, val, att_idx))
# =============================================================================

# =============================================================================
# root = 4
# subset1 = subsets[0]
# nbVal = len(m.attributes[4].values)+1
# print("nbVal: ", nbVal)
# for val in range(1, nbVal): #several values -> several subset
#     for att_idx in range(NB_ATTRIBUTES): #several attributes
#         if att_idx != root: #except the root
#             info_gain = dt.averageGain(subsets[ds_idx][val], m.attributes[att_idx])
#             firstfloor_gain[ds_idx].append((info_gain, val, att_idx))
# =============================================================================

# Full tree
tree1=dt.buildTree(dataset1, m.attributes, 2)
print('For the tree 1 :')
print("\n",dt.check(tree1, dataset1))
print("\n",dt.check(tree1, m.monk1test))

# =============================================================================
# tree2=dt.buildTree(datasets[1], m.attributes)
# print('For the tree 2 :')
# print("\n",dt.check(tree2, datasets[1]))
# print("\n",dt.check(tree2, m.monk2test))
# 
# tree3=dt.buildTree(datasets[2], m.attributes)
# print('For the tree 3 :')
# print("\n",dt.check(tree3, datasets[2]))
# print("\n",dt.check(tree3, m.monk3test))
# =============================================================================

#Dataset=dataset1+dataset2+dataset3
#TREE=d.buildTree(Dataset, m.attributes)
import drawtree_qt5 as DRW5

# DRW5.drawTree(tree1)
# DRW5.drawTree(tree2)
# DRW5.drawTree(tree3)



import random
import statistics as st

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def get_c(tref, cref, dataval):
    tree_list = dt.allPruned(tref)
    while True:
        switch = False
        for t in tree_list:
            c = dt.check(t, dataval)
            if c > cref:
                cref = c
                tref = t
                switch = True
        if switch == False:
            break
    return tref, cref

n = 10000
def repetition(dataset, n, fraction):
    c_list = []
    for i in range(n):
        datatrain, dataval = partition(dataset, fraction)
        tref = dt.buildTree(datatrain, m.attributes)
        cref = dt.check(tref, dataval)
        tref, cref = (get_c(tref, cref, dataval))
        c_list.append(cref)
    mean = st.mean(c_list)
    var = st.variance(c_list)
    return mean, var
    # return st.mean(c_list), st.variance(c_list)


frac = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
def drawMean(nbCourbes, ds, title):
    plt.figure()
    for step in range(nbCourbes):
        Ymean = []
        for f in frac:
            mean, _ = repetition(ds, n, f)
            Ymean.append(1-mean)
        plt.plot(frac, Ymean)
        plt.xlabel("Size of training set : Fraction of initial dataset")
        plt.ylabel("Test error")
        plt.title(title)
    plt.show()
    
def drawVar(nbCourbes, ds, title):
    plt.figure()    
    for step in range(nbCourbes):
        Yvar = []
        for f in frac:
            _, var = repetition(ds, n, f)
            Yvar.append(var)
        plt.plot(frac, Yvar)
        plt.xlabel("Size of training set : Fraction of initial dataset")
        plt.ylabel("Variance of testing dataset well classified fraction")
        plt.title(title)
    plt.show()

titleErr = "Classification error on the test sets - "
titleVar = "Classification variance on the test sets - "
drawMean(1, m.monk1, titleErr+"Monk1")
drawVar(1, m.monk1, titleVar+"Monk1")

drawMean(1, m.monk2, titleErr+"Monk2")
drawVar(1, m.monk2, titleVar+"Monk2")

drawMean(1, m.monk3, titleErr+"Monk3")
drawVar(1, m.monk3, titleVar+"Monk3")
















