# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 10:48:26 2022

@author: Cristina GH
"""

import argparse
import pandas as pd
import csv
import re
from textblob import TextBlob
import unidecode
import emoji
import nltk 
from nltk.corpus import stopwords
import stanza

nltk.download('stopwords')
nltk.download('punkt')

stanza.download('es', package='ancora', processors='tokenize, mwt, pos, lemma', verbose=True)
stNLP = stanza.Pipeline(processors='tokenize, mwt, pos, lemma', lang='es', use_gpu=True)



class TweetPreprocessing:
    
    def __init__(self, file):
        self.data = file
    
    def readFile(self):
        with open(self.data, 'r', encoding='utf-8') as csvfile:
            dialect = csv.Sniffer().sniff(csvfile.readline())
            try:            
                return pd.read_csv(self.data, sep=str(dialect.delimiter))
            except pd.errors.ParserError:
                return pd.read_csv(self.data, sep=str(dialect.delimiter), encoidng='utf-8', quoting=csv.QUOTE_NONE)
        
    def entityNormalization(tweet):        
        political_parties = ['podemos', 'partido popular', 'psoe', 'pp', 'ciudadanos', 'vox', 'ciu',
                              'unidas podemos', 'erc', 'esquerra republicana', 'eh bildu​', 'pnv', "c’s", "c's"]
    
        politicians = ['Pablo Iglesias', 'Pedro Sanchez', 'pedrosánchez', 'Pedro Sánchez', 'pedrosanchez', 'sánchez',
                        'pabloiglesias' 'Cayetana Álvarez de Toledo', 'Donald Trump', 'donaldtrump', 'Abascal', 'Santiago Abascal'
                        'Trump', 'Iglesias', 'Sanchez', 'Pablo Casado', 'Monasterio', 'Rocio Monasterio', 'Irene Montero',  'Montero',
                        'Albert Rivera', 'Rivera', 'albertrivera', 'pablocasado','Casado', 'Susana Díaz', 'Susana Diaz' 'Magallanes']
        
        fix_labels = ['\[political_party\]','\[\#political_party\]', '\[POLITICAL_PARTY\]']
        
        compile_parties = re.compile("|".join(political_parties), re.IGNORECASE)
        compile_politicians = re.compile("|".join(politicians), re.IGNORECASE)
        compile_labels = re.compile("|".join(fix_labels), re.IGNORECASE)
        
        tweet = re.sub(compile_parties, ' ', tweet.lower())
        tweet = re.sub(compile_politicians, ' ', tweet.lower())
        tweet = re.sub(compile_labels, 'politicalparty', tweet.lower())
    
        return tweet
    
    def basic_cleaning(tweet):       
        tweet = tweet.replace('[POLITICAL_PARTY]', ' ')
        tweet = tweet.replace('[#POLITICAL_PARTY]', ' ')
        # lowercase uppercase word
        tweet = re.sub(r'\b[A-Z]+(?:\s+[A-Z]+)*\b', lambda m: m.group(0).lower(), tweet)
        # separate_hashtag_words
        tweet = re.sub(r"([A-Z])|([0-9])", r" \1", tweet)
        # remove digits and hashtag char
        tweet = re.sub('\d+|#', ' ', tweet)
        tweet = re.sub('\d+[0-9]\.|(\d,)|(\d+)', ' ', tweet)
        # remove_extra_spaces
        tweet = re.sub(' +', ' ', tweet)
        # quoting as end of sent
        tweet = tweet.replace('"','.')
        tweet = tweet.replace("'",'.')
        tweet = tweet.strip()
        tweet = tweet.lstrip('.')
        # remove @user
        tweet = tweet.replace('@user', ' ')
        # double gender inflection
        tweet = tweet.replace('e/a', 'e')
        tweet = tweet.replace('os/as', 'os')
        tweet = tweet.replace('os/a', 'os')
        
        return tweet
    
    def lemmatize(tweet, categories=[]):        
        lemmatized_tweet = stNLP(str(tweet))
        pos_lemmas = [word.lemma + '|' + word.pos for sent in lemmatized_tweet.sentences for word in sent.words]
        
        if categories:
            good_lemmas = [lemma.split('|')[0] for lemma in pos_lemmas if lemma.split('|')[-1] in categories]
            return good_lemmas
        else:
            return pos_lemmas
    
    def removeStopwords(text):        
        text = ' '.join(text)
        blob = TextBlob(text).words
        custom_stopwords = ['daniel','hacer','decir','tener', 'q', 'user', 'tener', 'hacer', 'estar', 'ver', 'haber',
                              'dar', 'decir', 'querer', 'd', 'q', 'k', 'political_party', 'ir', 'vez', 'gran', 'mejor',
                            'preferir', 'saber', 'by', 'hoy', 'ayer','@user', 'ver', 'vs', 'hoy', 'ayer', 'mañana',
                            'solo', 'querer']
        remove_st = [word for word in blob 
                      if word not in stopwords.words('spanish')
                      and word not in custom_stopwords]
        return(' '.join(word for word in remove_st))
    
    def normalized_string(text):  
        text = unidecode.unidecode(text.lower().replace('ñ','ny'))
        text = text.replace('ny', 'ñ')
        return text
      
    def extract_emoji(text):      
        EMOJIS = emoji.UNICODE_EMOJI["en"]
        return ''.join([emoji for emoji in text if emoji in EMOJIS and emoji != str.strip(' 🏻')])
  
    def punct_signs(text):
        return re.sub('!|¡|\?|¿|\.|,|-|_|º|\\|/|\&|€|¬|#|\$|~|%|\(|\)|=|\+|\*|>|<|;|:|\{|\}|\[|\]|"|@', ' ', text)


def return_preprocessed_data(file, col, save_dir):
    '''
    

    Parameters
    ----------
    file : TYPE
        DESCRIPTION.
    col : TYPE
        DESCRIPTION.
    save_dir : TYPE
        DESCRIPTION.

    Returns
    -------
    df : TYPE
        DESCRIPTION.

    '''
    df = TweetPreprocessing(file).readFile()

    print('~ entity normalization')
    df['clean_data'] = df[col].apply(lambda x: TweetPreprocessing.entityNormalization(x))
    
    print('~ basic cleaning')
    df['clean_data'] = df['clean_data'].apply(lambda x: TweetPreprocessing.basic_cleaning(x))
    
    print('~ lemmatizing data')
    df['lemmatized_data'] = df['clean_data'].apply(lambda x: TweetPreprocessing.lemmatize(x, categories=['NOUN', 'VERB', 'ADV', 'ADJ']))
    
    print('~ normalizing text (no stw)')
    df['lemmatized_nostw'] = df['lemmatized_data'].apply(lambda x: TweetPreprocessing.normalized_string(' '.join(x)))
    
    print('~ removing stopwords')
    df['lemmatized_data'] = df['lemmatized_data'].apply(lambda x: TweetPreprocessing.removeStopwords(x))
    
    print('~ normalizing text (stw)')
    df['lemmatized_data'] = df['lemmatized_data'].apply(lambda x: TweetPreprocessing.normalized_string(x))
    
    print('~ removing punctuation signs on clean data')
    df['clean_data'] = df['clean_data'].apply(lambda x: TweetPreprocessing.punct_signs(x))
    
    print('~ extracting emojis')
    df['emojis'] = df[col].apply(lambda x: TweetPreprocessing.extract_emoji(x))

    #fname = save_dir + '\preprocessed_' + file.split('\\')[-1]
    fname = save_dir + '/preprocessed_' + file.split('/')[-1]
    print('file save at {fname}')
    df.to_csv(fname, sep='\t', encoding='utf-8', quoting=csv.QUOTE_NONE)
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-input_file', '--input_file', required=True, help="Path to input file") 
    parser.add_argument('-save_dir', '--save_dir', required=True, help="Path to saving directory") 
    
    args = parser.parse_args() 
    
    return_preprocessed_data(args.input_file, 'tweet', args.save_dir)
