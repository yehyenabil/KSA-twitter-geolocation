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


#data preporssing 

def stem(text,ps):   
    return " ".join([ps.stem(word) for word in text])

    
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("-d", "--dataset_size", type=str, help="Dataset size.", default='small',
                        choices=['micro', 'small', 'mid', 'large'])
    parser.add_argument("-c", "--classifier", type=str, help="Train a KSA State ",
                        default='state',)
    parser.add_argument("--max_words", type=int, help="Max number of words to analyze per user.", default=50)
    parser.add_argument("-v", "--vocab_size", type=int, help="Use the top N most frequent words.", default=50000)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch size.", default=512)
    parser.add_argument("--hidden_size", type=int, help="Number of neurons in the hidden layers.", default=128)
    parser.add_argument("--tensorboard", action="store_true", help="Track training progress using Tensorboard.",
                        default=True)
    args = parser.parse_args()

    environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    from twgeo.models import constants
    from twgeo.models.geomodel import Model

    if args.classifier == 'state':
        num_of_classes = 13
        x_train_b, y_train_b = read_csv_data('train.xlsx', 6,2 )
        df = pd.DataFrame (x_train_b, columns = ['TweetText'])
        df['ASCII'] = df['TweetText'].apply(lambda val: val.encode('utf-8').decode('unicode_escape',errors='ignore'))
        x_train_b2 = df['ASCII'].apply(lambda val: preprocesss_tweet(val))

        
        
        x_train=x_train_b2[12666:]
        y_train=y_train_b[12666:]
        
        
        
        x_dev=x_train_b2[:12665]
        y_dev=y_train_b[:12665]

        x_test1, y_test =read_csv_data('test.xlsx', 6,2 )
        df1 = pd.DataFrame (x_test1, columns = ['TweetText'])
        df1['ASCII'] = df1['TweetText'].apply(lambda val: val.encode('utf-8').decode('unicode_escape',errors='ignore'))
        x_test = df1['ASCII'].apply(lambda val: preprocesss_tweet(val))


        print("train.py: Training a KSA State classifier.")

    geoModel = Model(batch_size=args.batch_size, use_tensorboard=args.tensorboard)
    modelname="ssc"
    geomodel_state_model_file = path.join(constants.DATACACHE_DIR, 'geomodel_' + args.classifier + modelname)
    if path.exists(geomodel_state_model_file + ".h5"):
        print("Loading existing model at {0}".format(geomodel_state_model_file))
        geoModel.load_saved_model(geomodel_state_model_file)
    else:
        

        #you can choose any model do you like to apply from groModel and put his name it in l list below 
        l=["paper_model_MaxPooling"]
        for i in l:
            mycode='''geoModel.{}(empadding_wor_len=100,dropout_par=0.4,num_outputs=num_of_classes, time_steps=args.max_words, vocab_size=args.vocab_size,hidden_layer_size=args.hidden_size) '''
            exec(mycode.format(i))
            geoModel.train(x_train, y_train, x_dev, y_dev, epochs=args.epochs)
            geoModel.save_model(path.join(constants.DATACACHE_DIR, 'geomodel_' + args.classifier + i))
            
            
    
    gc.collect()
