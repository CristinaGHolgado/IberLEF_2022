import pandas as pd
import csv
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import unidecode
import re, time

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
import stanza

nltk.download('stopwords')
nltk.download('punkt')
stanza.download('es', package='ancora', processors='tokenize, mwt, pos, lemma', verbose=True)
stNLP = stanza.Pipeline(processors='tokenize, mwt, pos, lemma', lang='es', use_gpu=False)


# lazy text processing

def entity_normalization(tweet):
    #fix some entities
    political_parties = ['podemos', 'partido popular', 'psoe', 'pp', 'ciudadanos', 'vox', 'ciu',
                         'unidas podemos', 'erc', 'esquerra republicana', 'eh bildu​', 'pnv']

    politicians = ['Pablo Iglesias', 'Pedro Sanchez', 'pedrosánchez', 'Pedro Sánchez', 'pedrosanchez', 'sánchez',
                   'pabloiglesias' 'Cayetana Álvarez Toledo', 'Donald Trump', 'donaldtrump'
                   'Trump', 'Iglesias', 'Sanchez', 'Pablo Casado', 'Monasterio', 'Rocio Monasterio', 'Irene Montero',  'Montero',
                   'Albert Rivera', 'Rivera', 'albertrivera', 'pablocasado','Casado', 'Susana Díaz', 'Susana Diaz' 'Magallanes']
    fix_labels = ['\[political_party\]','\[\#political_party\]']
    
    compile_parties = re.compile("|".join(political_parties), re.IGNORECASE)
    compile_politicians = re.compile("|".join(politicians), re.IGNORECASE)
    compile_labels = re.compile("|".join(fix_labels), re.IGNORECASE)
    
    tweet = re.sub(compile_parties, 'politicalparty', tweet.lower())
    tweet = re.sub(compile_politicians, 'personpolitician', tweet.lower())
    tweet = re.sub(compile_labels, 'politicalparty', tweet.lower())

    return tweet

def basic_processing(tweet):
  remove_digits_hashtags = re.sub('\d+|#', ' ', tweet)
  remove_extra_spaces = re.sub(' +', ' ', remove_digits_hashtags)
  remove_ents = re.sub('political_party|person_politician', ' ', remove_extra_spaces)
  return remove_ents

def lemmatize(tweet, categories=[]):
  lemmatized_tweet = stNLP(str(tweet))
  pos_lemmas = [word.lemma + '|' + word.pos for sent in lemmatized_tweet.sentences for word in sent.words]
  good_lemmas = [lemma.split('|')[0] for lemma in pos_lemmas if lemma.split('|')[-1] in categories]
  return good_lemmas

def removeStopwords(text):
    text = ' '.join(text)
    blob = TextBlob(text).words
    custom_stopwords = ['daniel','hacer','decir','tener', 'q', '@user', 'user', 'tener', 'hacer', 'estar', 'ver',
                         'dar', 'decir', 'querer', 'd', 'q', 'k', 'political_party', 'ir', 'vez', 'gran', 'mejor',
                        'preferir', 'saber']
    remove_st = [word for word in blob 
                  if word not in stopwords.words('spanish')
                  and word not in custom_stopwords]
    return(' '.join(word for word in remove_st))

def normalized_string(text):
  return unidecode.unidecode(text.lower().replace('ñ','ny'))


t1 = time.time()


data = pd.read_csv("../data/development.csv") 
data = data.sort_values(by='tweet')

data['clean_data'] = data['tweet'].apply(lambda x: basic_processing(x))
data['clean_data'] = data['clean_data'].apply(lambda x: entity_normalization(x))
data['clean_data'] = data['clean_data'].apply(lambda x: lemmatize(x, categories=['NOUN', 'ADJ', 'VERB'])) # y adverbios?
data['clean_data'] = data['clean_data'].apply(lambda x: removeStopwords(x))
data['clean_data'] = data['clean_data'].apply(lambda x: normalized_string(x)) 


data['clean_data'] = data['clean_data'].apply(lambda x: re.sub('@user|politicalparty|politicalperson|politicalpar|anyo|personpolitician|personpoliticiar|gobierno|llegar|dejar', '', x))


pd.to_csv('~/Downloads/lemmatized_data.csv', sep='\t')


processed_data = data['clean_data'].values
tfidf = TfidfVectorizer(max_features=100, use_idf=True)
tdidf_matrix = tfidf.fit_transform(processed_data)
features = tfidf.get_feature_names()
terms = tfidf.vocabulary_


labels = data['ideology_binary']

pca = PCA(n_components = 3).fit(tdidf_matrix.toarray())
X_pca = pca.transform(tdidf_matrix.toarray())

sns.set(rc={"figure.figsize":(10, 6)})
sns.scatterplot(X_pca[:,0], X_pca[:, 1], hue=labels, legend='full', palette='Set1', s=10)

plt.show()

print('total time', (time.time()- t1), 'seconds')
