#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import string
import numpy as np
from halo import Halo
import pandas as pd
import math
import codecs
from datetime import datetime
import pickle
import matplotlib.pyplot as plt
import random
import re


# In[2]:


class Dataset():
    
    def __init__(self, x_filename, y_filename):
        self.x_filename = x_filename
        self.y_filename = y_filename
        self.x = []
        self.y = []
        self.wlen = 0
        self.wp = 0.5
        self.wn = 0.5
        self.datapoints = 0
        
    def readFiles(self):
        with codecs.open(self.x_filename, 'r', 'utf-8') as f:
            entities = f.read().split('\n')
            for entity in entities:
                if entity != '':
                    self.x.append(entity.strip())
        with codecs.open(self.y_filename, 'r', 'utf-8') as f:
            entity_labels = f.read().split('\n')
            for label in entity_labels:
                #print(label)
                try:
                    self.y.append(int(label))
                except Exception as e:
                    print(label)
        self.datapoints = len(self.y)
        self.wn = sum(self.y)/self.datapoints
        self.wp = 1 - self.wn
        self.wlen = len(self.x[0].split())
        


# In[3]:


class RandomIndexing(object):
    
    def __init__(self, X, filename, w_len, dimension=100, non_zero=20, non_zero_values=list([-1, 1]), left_window_size=0, right_window_size=3):
        self.__vocab = set()
        self.__dim = dimension
        self.__non_zero = non_zero
        self.__non_zero_values = non_zero_values
        self.__lws = left_window_size
        self.__rws = right_window_size
        self.__cv = None
        self.__rv = None
        self.X = X
        self.w_len = w_len
        self.filename = filename
        self.sentences = []
        
    def read_corpus(self):
        with codecs.open(self.filename, 'r', 'utf-8') as f:
            self.sentences = f.read().split('\n')
            for ind, sentence in enumerate(self.sentences):
                self.sentences[ind] = re.sub('[\.\,\(\)]', '', sentence)
        
    def read_vocabulary(self):
        print("Entered read vocab ",str(self.vocab_exists()))
        vocab_exists = self.vocab_exists()
        if vocab_exists:
            with open('vocab3.txt') as f:
                for line in f:
                    if(line.strip() == 'and'):
                        print('adding and to vocab')
                    if len(line.strip())>0:
                        self.__vocab.add(line.strip())
        self.__i2w = list(self.__vocab)
        return vocab_exists
    
        
    def write_vocabulary(self):
        for word in self.X:
            self.__vocab.add(word)
        file_name = 'vocab3.txt'
        with open(file_name, 'w') as f:
            for w in self.__vocab:
                f.write(w +'\n')
          
    def vocabulary_size(self):
        return len(self.__vocab)
    
    def vocab_exists(self):
        return os.path.exists('vocab3.txt')   
    
    def vectors_exist(self):
        return os.path.exists('ri_vectors3.txt')
    
    def create_word_vectors(self):
        
        self.__cv = {}
        self.__rv = {}
        for word in self.__vocab:
            rvlist = [0 for i in range(self.__dim)]
            randomindices = np.random.randint(0, self.__dim, self.__non_zero)
            for i in randomindices:
                rvlist[i] = np.random.choice(self.__non_zero_values)
            self.__rv[word] = np.array(rvlist)
            self.__cv[word] = np.array([0 for i in range(self.__dim)])
            if word == 'IL-2':
                print(self.__rv[word])
                print(self.__cv[word])
        
        for sentence in self.sentences:
            sent = sentence.split()
            words = []
            for ind, w in enumerate(sent):
                if ind+3 <= len(sent):
                    k1 = ''
                    for ik, k in enumerate(sent[ind:ind+3]):
                        if ik < len(sent[ind:ind+3])-1:
                            k1 += k + " "
                        else:
                            k1 += k
                    words.append(k1)
                else:
                    break
            for i, w in enumerate(words):
                #for i,w in enumerate(words):
                lwin = min(i, self.__lws)
                rwin = min(len(words)-1-i, self.__rws)
                left_random_sum = np.array([0 for i in range(self.__dim)])
                right_random_sum = np.array([0 for i in range(self.__dim)])
                if lwin > 0:
                    for left_index in range(lwin,0,-1):
                        try:
                            left_random_sum = left_random_sum + self.__rv[words[i-left_index]]
                        except Exception as e:
                            print(words[i-left_index])
                            continue
                if rwin > 0:
                    for right_index in range(rwin,0,-1):
                        try:
                            right_random_sum = right_random_sum + self.__rv[words[i+right_index]]
                        except Exception as e:
                            print(words[i+right_index])
                            continue
                try:
                    self.__cv[w] = self.__cv[w] + left_random_sum + right_random_sum
                except Exception as e:
                            print(w)
                            continue

                
    def get_word_vector(self, word):
        if word not in self.__vocab:
            return None
        else:
            return self.__cv[word]
      
                
    def train(self):
        
        if self.vocab_exists():
            print("Reading vocabulary...")
            self.read_vocabulary()
        else:
            print("Building vocabulary...")
            start = time.time()
            self.write_vocabulary()
            print("Built vocabulary in"+ str(round(time.time() - start, 2)) + "s.")

        if self.vectors_exist():
            print("Reading vectors...")
            self.read_word_vectors()
            #print(self.__cv['and'])
        else:
            print("Reading Data Corpus...")
            self.read_corpus()
            print(self.sentences[0])
            print("Creating vectors using random indexing...")
            start = time.time()
            self.create_word_vectors()
            self.write_vectors()
            
            print("Created random indexing vectors in "+ str(round(time.time() - start, 2))+"s")
        
    def train_and_persist(self):
        self.train()
        
    def get_embeddings(self, word):
        return self.__cv[word]
    
    def get_context_vectors(self):
        return self.__cv
    
    def get_all_embeddings(self):
        print(len(self.__cv))
        return list(self.__cv.values())
        
    def write_vectors(self):
        c = 0
        with open('ri_vectors3.txt', 'w', errors ='ignore') as f:
            for w in list(self.__cv.keys()):
                #if c == 10:
                 #   print(w + " " + " ".join(map(lambda x: "{0:.6f}".format(x), list(self.__cv[w]))) + "\n")
                f.write(w + " " + " ".join(map(lambda x: "{0:.6f}".format(x), list(self.__cv[w]))) + "\n")
                c += 1
       
                
    def read_word_vectors(self):
        cv = {}
        vectors_exist = self.vectors_exist()
        if vectors_exist:
            with open('ri_vectors3.txt') as f:
                for line in f:
                    vals = line.split()
                    word = vals[0]+" "+vals[1]+" "+vals[2]
                    cv[word] = np.array(list(map(float, vals[3:])))
        self.__cv = cv
        return vectors_exist
                


# In[4]:


class ManualIndexing(object):
    
    def __init__(self, filename):
        self.__cv = []
        self.filename = filename
        self.dim = 0
    
    def read_vectors(self):
        with open(self.filename) as f:
            for line in f:
                if len(line) > 0:
                    vals = line.split()
                    self.__cv.append(np.array(list(map(float, vals))))
        return self.__cv


# In[5]:


class BinaryLogisticRegression(object):
    
    #  ------------- Hyperparameters ------------------ #

    LEARNING_RATE = 0.01  # The learning rate.
    CONVERGENCE_MARGIN = 0.0001  # The convergence criterion.
    MAX_ITERATIONS = 20000 # Maximal number of passes through the datapoints in stochastic gradient descent.
    MINIBATCH_SIZE = 1000 # Minibatch size (only for minibatch gradient descent)

    # ----------------------------------------------------------------------
    
    def __init__(self, x=None, y=None, theta=None, wn =0.5, wp = 0.5):
        """
        Constructor. Imports the data and labels needed to build theta.

        @param x The input as a DATAPOINT*FEATURES array.
        @param y The labels as a DATAPOINT array.
        @param theta A ready-made model. (instead of x and y)
        """
        #if not any([x, y, theta]) or all([x, y, theta]):
         #   raise Exception('You have to either give x and y or theta')

        if theta is not None:
            self.FEATURES = len(theta)
            self.theta = theta

        elif x and y:
            # Number of datapoints.
            self.DATAPOINTS = len(x)

            # Number of features.
            self.FEATURES = len(x[0]) + 1

            # Encoding of the data points (as a DATAPOINTS x FEATURES size array).
            self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(x)), axis=1)

            # Correct labels for the datapoints.
            self.y = np.array(y)

            # The weights we want to learn in the training phase.
            self.theta = np.random.uniform(-1, 1, self.FEATURES)

            # The current gradient.
            self.gradient = np.zeros(self.FEATURES)
            
            #Cross-entropy Loss weights
            Nn = np.count_nonzero(self.y==0)
            Np = self.DATAPOINTS - Nn
            #self.wp = Nn/self.DATAPOINTS
            #self.wn = Np/self.DATAPOINTS
            self.wp = wp
            self.wn = wn
            
            
            
    def sigmoid(self, z):
        """
        The logistic function.
        """
        return 1.0 / ( 1 + np.exp(-z) )


    def conditional_prob(self, label, datapoint):
        """
        Computes the conditional probability P(label|datapoint)
        """
        prob = ((self.sigmoid(np.dot(self.theta, self.x[datapoint])))**label)*((1 - self.sigmoid(np.dot(self.theta, self.x[datapoint])))**(1-label))
        return prob


    def compute_gradient_minibatch(self, minibatch):
        """
        Computes the gradient based on the entire dataset
        (used for batch gradient descent).
        """
        sum_val = np.zeros(self.FEATURES)
        for i in minibatch:
            #sum_val += self.x[i]*(self.sigmoid(np.dot(self.theta, self.x[i])) - self.y[i])
            sum_val += self.x[i]*((self.wn * self.sigmoid(np.dot(self.theta, self.x[i])) * (1 - self.y[i])) - (self.wp * self.y[i] * (1 - self.sigmoid(np.dot(self.theta, self.x[i])))))
            
        self.gradient = (1/self.MINIBATCH_SIZE)*sum_val
        
        
        
    def fit(self):
        """
        Performs Mini-Batch Gradient Descent
        """
        #self.init_plot(self.FEATURES)
        counter = 0
        while(counter < self.MAX_ITERATIONS):
            
            #print(counter)
            mini_batch_points = np.random.randint(0, self.DATAPOINTS, self.MINIBATCH_SIZE)
            self.compute_gradient_minibatch(mini_batch_points)
            #self.update_plot(np.sum(np.square(self.gradient)))
            self.theta = self.theta - (self.LEARNING_RATE * self.gradient)
            if counter % 1000 == 0:
                print(counter)
                print(np.sum(self.gradient ** 2))
            if(np.sum(self.gradient ** 2) < self.CONVERGENCE_MARGIN):
                break
            counter += 1
        print("Fit complete in ",counter," iterations")
        
    def classify_input(self, x):
        """
        Classifies user datapoints
        """
        datapoints = len(x)
        predicted = []
        self.x = np.concatenate((np.ones((datapoints, 1)), np.array(x)), axis=1)
        for d in range(datapoints):
            prob = self.conditional_prob(1, d)
            predicted.append(1 if prob > .5 else 0)
        
        return predicted
        
        
    def classify_datapoints(self, test_data, test_labels):
        """
        Classifies datapoints
        """
        #print('Model parameters:');

        #print('  '.join('{:d}: {:.4f}'.format(k, self.theta[k]) for k in range(self.FEATURES)))
        #print(sum(self.theta))

        self.DATAPOINTS = len(test_data)

        self.x = np.concatenate((np.ones((self.DATAPOINTS, 1)), np.array(test_data)), axis=1)
        self.y = np.array(test_labels)
        confusion = np.zeros((self.FEATURES, self.FEATURES))
        for d in range(self.DATAPOINTS):
            prob = self.conditional_prob(1, d)
            predicted = 1 if prob > .5 else 0
            confusion[predicted][self.y[d]] += 1

        print('                       Real class')
        print('                 ', end='')
        print(' '.join('{:>8d}'.format(i) for i in range(2)))
        for i in range(2):
            if i == 0:
                print('Predicted class: {:2d} '.format(i), end='')
            else:
                print('                 {:2d} '.format(i), end='')
            print(' '.join('{:>8.3f}'.format(confusion[i][j]) for j in range(2)))


    def print_result(self):
        print(' '.join(['{:.2f}'.format(x) for x in self.theta]))
        print(' '.join(['{:.2f}'.format(x) for x in self.gradient]))
        
        
    def update_plot(self, *args):
        """
        Handles the plotting
        """
        if self.i == []:
            self.i = [0]
        else:
            self.i.append(self.i[-1] + 1)

        for index, val in enumerate(args):
            self.val[index].append(val)
            self.lines[index].set_xdata(self.i)
            self.lines[index].set_ydata(self.val[index])

        self.axes.set_xlim(0, max(self.i) * 1.5)
        self.axes.set_ylim(0, max(max(self.val)) * 1.5)

        plt.draw()
        plt.pause(1e-20)


    def init_plot(self, num_axes):
        """
        num_axes is the number of variables that should be plotted.
        """
        self.i = []
        self.val = []
        plt.ion()
        self.axes = plt.gca()
        self.lines =[]

        for i in range(num_axes):
            self.val.append([])
            self.lines.append([])
            self.lines[i], = self.axes.plot([], self.val[0], '-', c=[random.random() for _ in range(3)], linewidth=1.5, markersize=4)


# In[6]:


def evaluate(test_ds, model, randomindex):
    
    x = []
    print(len(test_ds.y))
    
    for test_word in test_ds.x:
        vector = randomindex.get_word_vector(test_word)
        if vector is None:
            #print("appending None")
            vector = np.array([0 for i in range(100)])
        x.append(vector)
    print(len(x))
    model.classify_datapoints(x, test_ds.y)
        


# In[7]:


def evaluate_manual(test_ds, model, manualindex):
    
    x = manualindex.read_vectors()
    model.classify_datapoints(x, test_ds.y)


# In[ ]:


if __name__ == '__main__':
    
    vector_flag = 'ri'
    
    train_ds = Dataset('data/X_words/training_X3_no_punctuation', 'data/Y_vectors/training_Y3_no_punctuation')
    train_ds.readFiles()
    test_ds = Dataset('data/X_words/testing_X3_no_punctuation', 'data/Y_vectors/testing_Y3_no_punctuation')
    test_ds.readFiles()
    
    if vector_flag == 'ri':
        ri = RandomIndexing(train_ds.x, 'data/corpus_text.txt',train_ds.wlen)
        ri.train()
        x = ri.get_all_embeddings()
        y = train_ds.y
        mod_name = 'b3.model'
        
    else:
        ri = ManualIndexing('data/X_feature_vectors_v2/training_X3_v2')
        x = ri.read_vectors()
        y = train_ds.y
        mod_name = 'b3_manual.model'
    print(x[0])    
    b = BinaryLogisticRegression(x, y, None,wn=train_ds.wn, wp=train_ds.wp)
    b.fit()
    
    dt = str(datetime.now()).replace(' ','_').replace(':','_').replace('.','_')
    newdir = 'model_' + dt
    os.mkdir( newdir )
    pickle.dump(b.theta, open(os.path.join(newdir, mod_name), 'wb'))
    
    print("Training Data Accuracy")
    b.classify_datapoints(x,y)
    print("Testing Data Accuracy")
    
    if vector_flag == 'ri':
        evaluate(test_ds, b, ri)
    else:
        ri = ManualIndexing('data/X_feature_vectors_v2/testing_X3_v2')
        evaluate_manual(test_ds, b, mi)
