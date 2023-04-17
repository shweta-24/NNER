#!/usr/bin/env python
# coding: utf-8

# In[12]:


from ClassificationModel1gram import RandomIndexing as ri1
from ClassificationModel1gram import ManualIndexing as mi1
from ClassificationModel1gram import BinaryLogisticRegression as blr1
from ClassificationModel2gram import RandomIndexing as ri2
from ClassificationModel2gram import ManualIndexing as mi2
from ClassificationModel2gram import BinaryLogisticRegression as blr21
from ClassificationModel3gram import RandomIndexing as ri3
from ClassificationModel3gram import ManualIndexing as mi3
from ClassificationModel3gram import BinaryLogisticRegression as blr31
import pickle
import numpy as np
from CreateVectors import Vectors


# In[14]:


mode = 'manual'


# In[15]:


if mode == 'ri':
    theta1 = pickle.load(open('model_2022-05-18_19_19_23_477034/b.model', 'rb'))
    theta2 = pickle.load(open('model_2022-05-18_19_23_19_866869/b2.model', 'rb'))
    theta3 = pickle.load(open('model_2022-05-18_19_36_12_387203/b3.model', 'rb'))
else:
    theta1 = pickle.load(open('model_2022-05-19_12_34_23_589091/b_manual.model', 'rb'))
    theta2 = pickle.load(open('model_2022-05-19_12_35_40_008350/b2_manual.model', 'rb'))
    theta3 = pickle.load(open('model_2022-05-19_12_38_11_106153/b3_manual.model', 'rb'))
    
b1 = blr1(None, None, theta1, 0.5, 0.5)
b2 = blr21(None, None, theta2, 0.5, 0.5)
b3 = blr31(None, None, theta3, 0.5, 0.5)


# In[16]:


if mode == 'ri':
    r1 = ri1(None, None, 1)
    r1.train()
    print("Got vectors for r1")
    r2 = ri2(None, None, 2)
    r2.train()
    print("Got vectors for r2")
    r3 = ri3(None, None, 3)
    r3.train()
    print("Got vectors for r3")
else:
    mi = Vectors()


# In[17]:


if mode == 'ri':
    while(True):
        entities = ''
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        x3 = []
        y3 = []
        entities_list1 = []
        entities_list2 = []
        entities_list3 = []
        entities = input(">")
        if entities == 'q':
            break

        entities_list = entities.split()
        entities_list1 = entities_list
        for entity in entities_list:
            if r1.get_word_vector(entity) is not None:
                x1.append(r1.get_word_vector(entity))
            else:
                x1.append(np.array([0 for i in range(100)]))
        y1 = b1.classify_input(x1)

        if len(entities_list) > 1:

            for ind, entity in enumerate(entities_list):
                if ind+2 <= len(entities_list):
                    k1 = ''
                    for ik, k in enumerate(entities_list[ind:ind+2]):
                        if ik < len(entities_list[ind:ind+2])-1:
                            k1 += k + " "
                        else:
                            k1 += k
                    entities_list2.append(k1)
                    if r2.get_word_vector(k1) is not None:
                        x2.append(r2.get_word_vector(k1))
                    else:
                        x2.append(np.array([0 for i in range(100)]))
                else:
                    break
            y2 = b2.classify_input(x2)

        if len(entities_list) > 2:

            for ind, entity in enumerate(entities_list):
                if ind+3 <= len(entities_list):
                    k1 = ''
                    for ik, k in enumerate(entities_list[ind:ind+3]):
                        if ik < len(entities_list[ind:ind+3])-1:
                            k1 += k + " "
                        else:
                            k1 += k
                    entities_list3.append(k1)
                    if r3.get_word_vector(k1) is not None:
                        x3.append(r3.get_word_vector(k1))
                    else:
                        x3.append(np.array([0 for i in range(100)]))
                else:
                    break
            y3 = b3.classify_input(x3)

        print(y1, end =" : ")
        print(entities_list1)
        print(y2, end =" : ")
        print(entities_list2)
        print(y3, end =" : ")
        print(entities_list3)
        if sum(y1) + sum(y2) + sum(y3) > 1:
            print("Contains nested named entities")
        elif sum(y1) + sum(y2) + sum(y3) > 0:
            print("Contains named entity")
        else:
            print("Not a named entity")
            
            
else:
    
    while(True):
        entities = ''
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        x3 = []
        y3 = []
        entities_list1 = []
        entities_list2 = []
        entities_list3 = []
        entities = input(">")
        if entities == 'q':
            break

        entities_list = entities.split()
        entities_list1 = entities_list
        for entity in entities_list:
            x1.append(mi.get_vector(entity, True, ''))
                
        y1 = b1.classify_input(x1)

        if len(entities_list) > 1:

            for ind, entity in enumerate(entities_list):
                if ind+2 <= len(entities_list):
                    k1 = ''
                    for ik, k in enumerate(entities_list[ind:ind+2]):
                        if ik < len(entities_list[ind:ind+2])-1:
                            k1 += k + " "
                        else:
                            k1 += k
                    entities_list2.append(k1)
                    x2.append(mi.get_multiword_vector(k1.split(), True, ''))
                else:
                    break
            y2 = b2.classify_input(x2)

        if len(entities_list) > 2:

            for ind, entity in enumerate(entities_list):
                if ind+3 <= len(entities_list):
                    k1 = ''
                    for ik, k in enumerate(entities_list[ind:ind+3]):
                        if ik < len(entities_list[ind:ind+3])-1:
                            k1 += k + " "
                        else:
                            k1 += k
                    entities_list3.append(k1)
                    x3.append(mi.get_multiword_vector(k1.split(), True, ''))
                else:
                    break
            y3 = b3.classify_input(x3)

        print(y1, end =" : ")
        print(entities_list1)
        print(y2, end =" : ")
        print(entities_list2)
        print(y3, end =" : ")
        print(entities_list3)
        if sum(y1) + sum(y2) + sum(y3) > 1:
            print("Contains nested named entities")
        elif sum(y1) + sum(y2) + sum(y3) > 0:
            print("Contains named entity")
        else:
            print("Not a named entity")

