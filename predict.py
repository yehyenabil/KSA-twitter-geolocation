import math
import re
import time

import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import argparse
import math
import re
import time
from keras.preprocessing.text import Tokenizer
import gc
import numpy as np
import nltk
import pandas as pd
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from os import path, environ

from os import path, environ
def stem(text,ps):   
    return " ".join([ps.stem(word) for word in text])
 
 #print(n)


    
def remove_emoji(string):    
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002500-\U00002BEF"  # chinese char
                           u"\U00002702-\U000027B0"
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\U0001f926-\U0001f937"
                           u"\U00010000-\U0010ffff"
                           u"\u2640-\u2642"
                           u"\u2600-\u2B55"
                           u"\u200d"
                           u"\u23cf"
                           u"\u23e9"
                           u"\u231a"
                           u"\ufe0f"  # dingbats
                           u"\u3030"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)
def preprocesss_tweet(x):
    
    import re
    import string 
    from nltk.stem import PorterStemmer
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
     
     
     # removing
    x = re.sub(r'http(?s).*', '', x)
    x= re.sub(r'[ا-ي]', "", x)
    x=re.sub("_" , '',x)
    x=re.sub('@[A-Za-z0-9]+','',x) 
    x= re.sub("\d", "",x)
    x= re.sub('#[ا-ي]+', "", x)
    x= re.sub('[ا-ي]#+', "", x)
    x= re.sub("أ","", x)
    x=re.sub('#[A-Za-z0-9]+','',x) 
    x=re.sub('ؤ','',x)
    #print(x)
    
    # remove punctuations
    string.punctuation
    j=''.join([c for c in x if c not in string.punctuation])

     #print(j)    
    #remove emojii
    z=remove_emoji(j)
     #print(z)    
     #remove stopwords
    from nltk.corpus import stopwords
    stopwords=set(stopwords.words('english'))
    z=" ".join([b for b in z.split()if b not in stopwords])

     #print(z)
    # tokenize
    tokens=word_tokenize(z)
    #print(tokens)
    #stemmer
    ps = PorterStemmer()
    words = (ps.stem(re.sub('(.)\\1{2,}', '\\1\\1', w)).lower() for w in tokens)
    tweet_txt = ' '.join(words)
    return (tweet_txt)

def read_csv_data(csv_filename: str, location_column_idx: int, tweet_txt_column_idx: int):
    """
    Pre-process raw tweet data from a csv file. For each row, this function will:

        1. Tokenize the tweet text.
        2. Limit repeated characters to a maximum of 2. For example: 'Greeeeeetings' becomes 'Greetings'.
        3. Perform `Porter stemming  <https://en.wikipedia.org/wiki/Stemming>`_ on each token.
        4. Convert each token to lower case.

    :param csv_filename:
    :param location_column_idx: The zero-based index of the CSV column that contains the location information.
            The data itself must be a discrete value (a string or integer).
    :param tweet_txt_column_idx: The zero-based index of the CSV column that contains the tweet text.
    :return: Tuple (preprocessed_tweets, locations)
    """

    print("Parsing data from {0}...".format(csv_filename))
    df = pd.read_excel(csv_filename)
    df = df.sample(frac=1).reset_index(drop=True)
    df.drop(df[(df['state'] =='state')].index, inplace=True)
    tweets = df.iloc[:, tweet_txt_column_idx ].values
    locations = df.iloc[:, location_column_idx ].values

    ps = PorterStemmer()

    total_lines = len(tweets)
    percent_pt = math.ceil(total_lines / 500)
    now = time.time()
    start = now

    for i in range(0, len(tweets)):
        if (i % percent_pt == 0):
            now = time.time()
            if i != 0:
                time_per_unit = (now - start) / i
            else:
                time_per_unit = 999
            eta = time_per_unit * (total_lines - i)
            if eta > 3600:
                eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)
            elif eta > 60:
                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:
                eta_format = '%ds' % eta

            info = "\r{0:.2f}% complete. ({1:,}/{2:,}) ETA: {3}        ".format(i / percent_pt / 5, i, total_lines,
                                                                                eta_format)
            print(info, end='')

        tweet_txt = str(tweets[i])  
        words = word_tokenize(tweet_txt)

        words = (ps.stem(re.sub('(.)\\1{2,}', '\\1\\1', w)).lower() for w in words)
        tweet_txt = ' '.join(words)
        tweets[i] = tweet_txt
    print("\r100% complete...")
    return tweets, locations
from collections import Counter
if __name__ == '__main__':
    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from twgeo.models.geomodel import Model
    from twgeo.models import  constants


    x_test1, y_test =read_csv_data('test.xlsx', 6,2  )
    df1 = pd.DataFrame (x_test1, columns = ['TweetText'])
    df1['ASCII'] = df1['TweetText'].apply(lambda val: val.encode('utf-8').decode('unicode_escape',errors='ignore'))
    x_test = df1['ASCII'].apply(lambda val: preprocesss_tweet(val)) 

    geoModel = Model(batch_size=512)

    geoModel.load_saved_model(path.join(constants.DATACACHE_DIR, 'geomodel_state'+"paper_model_MaxPooling"))
    print(x_test[10000])
    print(y_test[10000])
    predictions = geoModel.predict(x_test[10000])
    print(predictions)
    c = Counter(predictions)
    print(c.most_common(1))

    

