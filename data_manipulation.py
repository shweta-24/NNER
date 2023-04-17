#!/usr/bin/env python
# coding: utf-8

# In[1]:


from bs4 import BeautifulSoup 
import pandas as pd
import re

from CreateVectors import Vectors


# In[2]:


# Read data from file

data_file = "GENIA_term_3.02/GENIAcorpus3.02.xml"

with open(data_file, 'r') as in_file:
    contents = in_file.read()


soup = BeautifulSoup(contents,"xml")

articles = soup.find_all("article")

training_data = articles[:1600]
testing_data = articles[1600:]


# In[3]:


# Save array to text file
def write_text_to_file(array, file):
    with open(file, "w") as out_file:
        for row in array:
            out_file.write(str(row))
            out_file.write('\n')
#write_text_to_file(sentences_text, output_text_file) #Uncomment if the a new text file should be created


# In[4]:


# Get all sentences from data (first return is sentences with tags, second return cleaned sentences)
def get_sentences(data):
    sentences = []
    sentences_text = [] # The cleaned text without tags
    for article in data:
        sentences_in_article = article.find_all("sentence")
        for sentence in sentences_in_article:
            sentences.append(sentence)
            sentence_text = sentence.get_text()
            sentence_text = re.sub('[\.\,\(\)]', '', sentence_text)
            sentences_text.append(sentence_text)
    return sentences, sentences_text

_, training_text = get_sentences(training_data)
_, testing_text = get_sentences(testing_data)


# In[5]:


# Creates list with all named entities from each sentence
def get_entity_names(sentences):
    names = []
    for sentence in sentences:
        cons = sentence.find_all("cons")
        sentence_cons = []
        for con in cons:
            sentence_cons.append(con.get_text())
        names.append(sentence_cons)
    return names


# In[6]:


# Returns vectors to use for training ML-models
# @param ngram - decides how many words to use for each input (each "name")
# @return two vectors X and Y
# X - a vector with each word (or ngram) from the corpus
# Y - a corresponding vector with a 1 if the word is a named entity, 0 if it's not
def get_name_vectors(sentences, names, ngram):
    X = []
    Y = []
    
    assert len(sentences) == len(names)
    
    count = 0
    
    for i in range(len(sentences)):
        text = sentences[i].split()
                
        for j in range(len(text)-ngram+1):
            
            sub = ' '.join(text[j:j+ngram])
            
            X.append(sub)
            
            if sub in names[i]:
                Y.append(1)
            else:
                Y.append(0)
                
    return X,Y


# In[11]:


# Run cell to get X and Y vectors both training and testing data
# (X vectors are only the n-grams in text, not word embeddings)

training_sentences_raw, training_sentences = get_sentences(training_data)
testing_sentences_raw, testing_sentences = get_sentences(testing_data)

training_names = get_entity_names(training_sentences_raw)
testing_names = get_entity_names(testing_sentences_raw)


trainX1, trainY1 = get_name_vectors(training_sentences, training_names, 1)
trainX2, trainY2 = get_name_vectors(training_sentences, training_names, 2)
trainX3, trainY3 = get_name_vectors(training_sentences, training_names, 3)

testX1, testY1 = get_name_vectors(testing_sentences, testing_names, 1)
testX2, testY2 = get_name_vectors(testing_sentences, testing_names, 2)
testX3, testY3 = get_name_vectors(testing_sentences, testing_names, 3)


# In[19]:


# Run cell to create feature vectors and write them to file

vic = Vectors()
vic.create_vectors(training_text,3)
vic.write_vectors_to_file("data/X_feature_vectors_v2/training_X3_v2")


# In[19]:


# Run cell to write training and testing data to file

write_text_to_file(trainX1, "data/training_X1_no_punctuation")
write_text_to_file(trainX2, "data/training_X2_no_punctuation")
write_text_to_file(trainX3, "data/training_X3_no_punctuation")

write_text_to_file(trainY1, "data/training_Y1_no_punctuation")
write_text_to_file(trainY2, "data/training_Y2_no_punctuation")
write_text_to_file(trainY3, "data/training_Y3_no_punctuation")



write_text_to_file(testX1, "data/testing_X1_no_punctuation")
write_text_to_file(testX2, "data/testing_X2_no_punctuation")
write_text_to_file(testX3, "data/testing_X3_no_punctuation")

write_text_to_file(testY1, "data/testing_Y1_no_punctuation")
write_text_to_file(testY2, "data/testing_Y2_no_punctuation")
write_text_to_file(testY3, "data/testing_Y3_no_punctuation")

