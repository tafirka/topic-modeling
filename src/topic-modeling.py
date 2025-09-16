'''
Author: Ismail DIOP
Date: May 2019

'''

# - IMPORT PACKAGES

import re
import numpy as np
import pandas as pd
from pprint import pprint

# nltk
import nltk

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel, Phrases
from gensim.models.phrases import Phraser

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
#%matplotlib inline

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

# --------------------------------------------------------

# - PREPARE STOPWORDS

# NLTK Stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('french')
stop_words.extend(['après', 'près', 'les', 'dès','tous', 'sous', 'autre', 'autres', 'sans', 'comme', \
                       'parmi', 'quoi', 'où', 'vers', 'voici', 'quant', 'dont', 'ici', 'avoir', 'si',\
                       'tant', 'entre', 'surtout', 'dirai', 'toute','chers','chacun','auquel', 'lors',\
                       'ferai', 'auto', 'avant', 'afin', 'ainsi', 'frère', 'ould', 'aziz', 'frère', 'autour',\
                       'grand', 'faire', 'sœur', 'chère', 'plus', 'dire', 'donc', 'travers', 'cher', 'pay', 'encore',\
                       'chose', 'dessus', 'pensaient', 'oui', 'an', 'ans', 'précédent', 'tel', 'cas', 'car',\
                       'être', 'tout', 'peux', 'toutefois', 'mohamed', 'puisse', 'puisse', 'cet', 'ils', 'duquel', 'chères', 'sœurs', 'frères', 'tout'])

                       # , '', '', '', '', '', '', '', '', ''

#print(stop_words)

# --------------------------------------------------------


#load dataset
'''
filename = 'data/MOCMAG.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
'''

# - TOKENIZE WORDS AND CLEAN-UP TEXT

#split into sentences
#sentences = nltk.sent_tokenize(text)
sentences = [line for line in open('data/MOM.txt', encoding='utf-8')]
#print(sentences[0:4])


def sent_to_words(sentences):
    for sentence in sentences:
        yield(simple_preprocess(str(sentence), deacc=False))  # deacc=True removes punctuations

data_words = list(sent_to_words(sentences))

#print(data_words[:5])

# -----------------------------------------------------


# - CREATING BIGRAM AND TRIGRAM MODELS

# Build the bigram and trigram models
bigram = Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = Phrases(bigram[data_words], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = Phraser(bigram)
trigram_mod = Phraser(trigram)

# See trigram example
#print(trigram_mod[bigram_mod[data_words[0]]])

# -----------------------------------------------------

# - REMOVE STOPWORDS, MAKE BIGRAMS AND LEMMATIZE

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]


def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# let's call the functions in order

# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
# -- data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'fr' model, keeping only tagger component (for efficiency)
# python3 -m spacy download fr
nlp = spacy.load('fr', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
# -- data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#print(data_lemmatized[:5])

# -----------------------------------------------------

# - CREATE THE DICTIONARY AND CORPUS NEEDED FOR TOPIC MODELING

# Create Dictionary
id2word = corpora.Dictionary(data_words_nostops)

# Create Corpus
texts = data_words_nostops

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# View
#print(corpus[:1])
#print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])


# -----------------------------------------------------

# - BUILDING THE TOPIC MODEL

# Build LDA model
'''
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=5,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


'''

# -----------------------------------------------------

# -  VIEW THE TOPICS IN LDA MODEL

# Print the Keyword in the 10 topics
# -- # pprint(lda_model.print_topics())

# -- doc_lda = lda_model[corpus]


# -----------------------------------------------------

# - COMPUTE MODEL PERPLEXITY AND COHERENCE SCORE


# Compute Perplexity
# -- print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
# -- coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_nostops, dictionary=id2word, coherence='c_v')
# -- coherence_lda = coherence_model_lda.get_coherence()

# -- print('\nCoherence Score: ', coherence_lda)


# -----------------------------------------------------

# - VISUALIZE THE TOPICS-KEYWORDS

# Visualize the topics

# -- pyLDAvis.enable_notebook()
# -- vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
# -- vis

# -----------------------------------------------------

# - BUILDNG LDA MALLET MODEL

'''
# Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
mallet_path = r'mallet/mallet-2.0.8/bin/mallet'
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=10, id2word=id2word)

# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_words_nostops, dictionary=id2word, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)

'''

# -----------------------------------------------------

# -  FIND THE OPTIMAL NUMBER OF TOPICS FOR LDA

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        #model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus, num_topics=num_topics, id2word=id2word)
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=num_topics,
                                           random_state=100,
                                           update_every=1,
                                           chunksize=100,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# Can take a long time to run.
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=data_words_nostops, start=2, limit=20, step=3)

# -- limit=20; start=2; step=3;
# -- x = range(start, limit, step)

# Print the coherence scores
# -- for m, cv in zip(x, coherence_values):
# --    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))

# Select the model and print the topics
optimal_model = model_list[4]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))











