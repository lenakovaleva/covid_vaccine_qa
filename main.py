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

# Opening csv
df = pd.read_csv('questions.csv', encoding='utf-8')
lst = []
stops = set(stopwords.words("russian"))
morph = pymorphy2.MorphAnalyzer()

# Cleaning the data
def clean_question(sent):
    cleaned_line = sent.strip("?'")
    cleaned_line = re.sub(r"[\'\"\?,\.!\(\)\s{2,}]", " ", cleaned_line)

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
#
# model = FastText()
# # model.build_vocab(corpus_file=corpus_file)
# model.build_vocab(sentences=MyIter())
# model.train(
#     corpus_file=corpus_file, epochs=model.epochs,
#     total_examples=model.corpus_count, total_words=model.corpus_total_words,
# )
# model.save(model)
#
# wv = model.wv
# say_vector = model.wv['ночь']

# # Retrieve all of the vectors from a trained model
wv = KeyedVectors.load('doc2vec.wordvectors') # load the word vectors model
# vector = wv['сегодня']

sentence_vectors = []
for line in cleaned_questions:
    sum_of_words = 0
    for word in line:
        sum_of_words += wv[word]
    sentence_vector_list = (sum_of_words / len(line))
    sentence_vectors.append(sentence_vector_list)
df['Sentence_Vector'] = pd.Series(sentence_vectors)


# # Get the vector of the user questions
# input_question = input()
input_question = 'Где привиться от ковид?'
# input_question = 'Заболел ребенок'
# input_question = 'Где растут кокосы?'
# input_question = 'Почему лучше привиться, чем переболеть?'
# input_question = 'Почему в России объявлена массовая вакцинация против коронавирусной инфекции нового типа (COVID-19)?'
# input_question = 'Больной ковидом заразен в течение 90 дней?'

cleaned_user_question = clean_question(input_question)

# # Get the vector of the words from user question
user_sentence_vector = []
sum_of_words = 0
for word in cleaned_user_question:
    sum_of_words += wv[word]
user_sentence_vector = sum_of_words / len(cleaned_user_question)

# Cosine distance
normalized_vectors = sentence_vectors / np.linalg.norm(sentence_vectors)
normalized_user_sentence_vector = user_sentence_vector / np.linalg.norm(user_sentence_vector)

ls_cos_sim = []
for sent in sentence_vectors:
    ls_cos_sim.append(1 - np.dot(sent / np.linalg.norm(sent), normalized_user_sentence_vector))
min_cos_sim = min(ls_cos_sim)
index = ls_cos_sim.index(min_cos_sim)
print(df.loc[[index], ['Question']])