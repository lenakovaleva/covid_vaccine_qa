#Import all the dependencies
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import pymorphy2
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import FastText
from gensim.models import KeyedVectors
import logging
from telegram import Update, ForceReply
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, CallbackContext

from flask import Flask,request
import requests
import sys

# Opening csv
df = pd.read_csv('questions.csv', encoding='utf-8')
lst = []
stops = set(stopwords.words("russian"))
morph = pymorphy2.MorphAnalyzer()
print(df)

# Get the questions from csv
questions = df['Question'].squeeze()
ls_questions = questions.to_list() # почистить кавычки
answers = df['Answer'].squeeze()
ls_answers = answers.to_list() # почистить кавычки

# Cleaning the data
def clean_question(sent):
    cleaned_line = sent.strip("?'")
    cleaned_line = re.sub(r"[\'\"\»\?,\.!\(\)\s{2,}]", " ", cleaned_line)

    words = [w for w in cleaned_line.lower().split() if not w in stops]

    tokens = []
    for token in words:
        tokens.append(morph.parse(token)[0].normal_form)
    return tokens

cleaned_questions = []
for sent in df.Question:
    cleaned_questions.append(clean_question(sent))

# # Opening the trained model
# model = FastText.load_fasttext_format(model_file='D:\ML\COVID_QA\chatbot_v1\cc.ru.300.bin\cc.ru.300.bin')
#
# # Testing the model
# print(model.wv.most_similar('сегодня', topn=10))
# print(model.wv.similarity('утро', 'ночь')) #0.5608701
# say_vector = model.wv['ночь']
# print(say_vector)

# # Store just the words + their trained embeddings.
# word_vectors = model.wv
# word_vectors.save("D:\ML\COVID_QA\chatbot_v1\cc.ru.300.bin\cc.ru.300.bin")

# # Load back with memory-mapping = read-only, shared across processes.
# wv = KeyedVectors.load("D:\ML\COVID_QA\chatbot_v1\cc.ru.300.bin\cc.ru.300.bin", mmap='r')
#
# vector = wv['сегодня']  # Get numpy vector of a word
# print(model.wv.vocab.keys())
#
# model = FastText()
# # model.build_vocab(corpus_file=corpus_file)
# model.build_vocab(questions=MyIter())
# model.train(
#     corpus_file=corpus_file, epochs=model.epochs,
#     total_examples=model.corpus_count, total_words=model.corpus_total_words,
# )
# model.save(model)
#
# wv = model.wv
# print(wv)
# print(wv['ночь'])
# say_vector = model.wv['ночь']


# # Retrieve all of the vectors from a trained model
# print(wv.vectors_vocab)

wv = KeyedVectors.load('doc2vec.wordvectors') # load the word vectors model
# vector = wv['сегодня']
# print(vector)
# print(type(wv)) # <class 'gensim.models.keyedvectors.FastTextKeyedVectors'>

sentence_vectors = []
for line in cleaned_questions:
    sum_of_words = 0
    for word in line:
        sum_of_words += wv[word]
    sentence_vector_list = (sum_of_words / len(line))
    sentence_vectors.append(sentence_vector_list)
df['Sentence_Vector'] = pd.Series(sentence_vectors)

# # Get the vector of the words from user question
def user_sent_prep(input_question):
    cleaned_user_question = clean_question(input_question)
    user_sentence_vector = []
    sum_of_words = 0
    for word in cleaned_user_question:
        sum_of_words += wv[word]
    user_sentence_vector = sum_of_words / len(cleaned_user_question)
    return user_sentence_vector

# Find the most relevant question form the database
def cos_sim(user_sentence_vector):
    normalized_vectors = sentence_vectors / np.linalg.norm(sentence_vectors)
    normalized_user_sentence_vector = user_sentence_vector / np.linalg.norm(user_sentence_vector)

    ls_cos_sim = []
    for sent in sentence_vectors:
        ls_cos_sim.append(1 - np.dot(sent / np.linalg.norm(sent), normalized_user_sentence_vector))
    min_cos_sim = min(ls_cos_sim)
    index = ls_cos_sim.index(min_cos_sim)
    return index

def cos_sim_sorted(user_sentence_vector):
    normalized_vectors = sentence_vectors / np.linalg.norm(sentence_vectors)
    normalized_user_sentence_vector = user_sentence_vector / np.linalg.norm(user_sentence_vector)

    ls_cos_sim = []
    for sent in sentence_vectors:
        ls_cos_sim.append(1 - np.dot(sent / np.linalg.norm(sent), normalized_user_sentence_vector))
    ls_cos_sim_sorted = []
    ls_cos_sim_sorted = sorted(ls_cos_sim)
    return ls_cos_sim_sorted