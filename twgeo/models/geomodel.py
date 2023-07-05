import pickle
import time
from os import path, environ
from keras.preprocessing.text import Tokenizer
import gc
import numpy as np
import pandas as pd
import keras

from keras.layers import Bidirectional,Dense, Dropout, LSTM, Embedding ,GlobalMaxPooling1D,TimeDistributed,Flatten ,Conv1D,Input,concatenate
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from sklearn import preprocessing




def top_5_acc(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=5)


environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Model:
    

    def loadEmbeddingMatrix(self,empadding_wor_len,typeToLoad):
        embed_size=0
        train = pd.read_excel('train.xlsx')
        train.drop(train[(train['state'] =='state')].index, inplace=True)
        train["TweetText "] =train["TweetText "].astype(str)
        list_sentences_train = train["TweetText "]
        print(list_sentences_train[0])
        max_features = empadding_wor_len
        tokenizer = Tokenizer(num_words=max_features)
        tokenizer.fit_on_texts(list(list_sentences_train))
        
        
        
        #load different embedding file from Kaggle depending on which embedding 
        #matrix we are going to experiment with
        if(typeToLoad=="glove"):
            path="glove.6B.{}d.txt".format(empadding_wor_len)
            EMBEDDING_FILE =path
            embed_size = empadding_wor_len
        #elif(typeToLoad=="word2vec"):
            #word2vecDict = word2vec.KeyedVectors.load_word2vec_format("../input/googlenewsvectorsnegative300/GoogleNews-vectors-negative300.bin", binary=True)
            #embed_size = 300
        elif(typeToLoad=="fasttext"):
            EMBEDDING_FILE='../input/fasttext/wiki.simple.vec'
            embed_size = empadding_wor_len
        
        if(typeToLoad=="glove" or typeToLoad=="fasttext" ):
            embeddings_index = dict()
            #Transfer the embedding weights into a dictionary by iterating through every line of the file.
            f = open(EMBEDDING_FILE,encoding="utf8")
            for line in f:
                #split up line into an indexed array
                values = line.split()
                #first index is word
                word = values[0]
                #store the rest of the values in the array as a new array
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs #50 dimensions
            f.close()
            print('Loaded %s word vectors.' % len(embeddings_index))
        else:
            embeddings_index = dict()
            #for word in word2vecDict.wv.vocab:
                #embeddings_index[word] = word2vecDict.word_vec(word)
            print('Loaded %s word vectors.' % len(embeddings_index))
            
        gc.collect()
        #We get the mean and standard deviation of the embedding weights so that we could maintain the 
        #same statistics for the rest of our own random generated weights. 
        all_embs = np.stack(list(embeddings_index.values()))
        emb_mean,emb_std = all_embs.mean(), all_embs.std()
        
        nb_words = len(tokenizer.word_index)
        #We are going to set the embedding size to the pretrained dimension as we are replicating it.
        #the size will be Number of Words in Vocab X Embedding Size
        embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
        gc.collect()
        
        #With the newly created embedding matrix, we'll fill it up with the words that we have in both 
        #our own dictionary and loaded pretrained embedding. 
        embeddedCount = 0
        for word, i in tokenizer.word_index.items():
            i-=1
            #then we see if this word is in glove's dictionary, if yes, get the corresponding weights
            embedding_vector = embeddings_index.get(word)
            #and store inside the embedding matrix that we will train later on.
            if embedding_vector is not None: 
                embedding_matrix[i] = embedding_vector
                embeddedCount+=1
        print('total embedded:',embeddedCount,'common words')
        
        del(embeddings_index)
        gc.collect()
        
        #finally, return the embedding matrix
        return embedding_matrix

    def __init__(self, use_tensorboard=True, batch_size=512):
        """

        :param use_tensorboard: Track training progress using Tensorboard. Default: true.
        :param batch_size: Default: 64
        """
        self._use_tensorboard = use_tensorboard
        self._batch_size = batch_size
        self._tokenizer = None
        self._label_encoder = None


    def base_model(self, empadding_wor_len,dropout_par,num_outputs, time_steps=500, vocab_size=20000, hidden_layer_size=128):
        """
         model Architecture:
             US model 
        """
        print("base_model")

        self._time_steps = time_steps
        self._vocab_size = vocab_size
        self._num_outputs = num_outputs
        self._hidden_layer_size = hidden_layer_size
        self.empadding_wor_len = empadding_wor_len
        self.dropout_par = dropout_par



        print("\nBuilding model...\nHidden layer size: {0}\nAnalyzing up to {1} words for each sample.".format(
            hidden_layer_size, time_steps))
        


        self._model = Sequential()
        self._model.add(Embedding(self._vocab_size, empadding_wor_len))
        self._model.add(LSTM(hidden_layer_size, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
        self._model.add(LSTM(hidden_layer_size, dropout=0.5, recurrent_dropout=0.5))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(self._num_outputs, activation='softmax'))
        self._model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy', top_5_acc])
        
    def base_model_glove(self, empadding_wor_len,dropout_par,num_outputs, time_steps=500, vocab_size=20000, hidden_layer_size=128):
        """
         model Architecture:
             US model 
        """

        print("base_model_glove")


        self._time_steps = time_steps
        self._vocab_size = vocab_size
        self._num_outputs = num_outputs
        self._hidden_layer_size = hidden_layer_size
        self.empadding_wor_len = empadding_wor_len
        self.dropout_par = dropout_par
        
        embedding_matrix=self.loadEmbeddingMatrix(empadding_wor_len,"glove")
        print(embedding_matrix.shape)

        print("\nBuilding model...\nHidden layer size: {0}\nAnalyzing up to {1} words for each sample.".format(
            hidden_layer_size, time_steps))
        

        self._model = Sequential()
        self._model.add(Embedding(self._vocab_size,output_dim=empadding_wor_len,weights=[embedding_matrix],input_length=empadding_wor_len,trainable=False))        
        self._model.add(LSTM(hidden_layer_size, dropout=0.5, recurrent_dropout=0.5, return_sequences=True))
        self._model.add(LSTM(hidden_layer_size, dropout=0.5, recurrent_dropout=0.5))
        self._model.add(Dropout(0.5))
        self._model.add(Dense(self._num_outputs, activation='softmax'))
        self._model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy', top_5_acc])
        

        

        
      

    def paper_model_Bidirectional(self,empadding_wor_len ,dropout_par ,num_outputs, time_steps=500, vocab_size=20000, hidden_layer_size=128):
        """
         model Architecture:
             add Bidirectional LSTM

        """
        print("paper_model_Bidirectional")

        self._time_steps = time_steps
        self._vocab_size = vocab_size
        self._num_outputs = num_outputs
        self._hidden_layer_size = hidden_layer_size
        self.empadding_wor_len=empadding_wor_len
        self.dropout_par=dropout_par

        print("\nBuilding model...\nHidden layer size: {0}\nAnalyzing up to {1} words for each sample.".format(
            hidden_layer_size, time_steps))
        
        self._model = Sequential()
        self._model.add(Embedding(self._vocab_size, empadding_wor_len))
        self._model.add(Bidirectional(LSTM(hidden_layer_size, dropout=0.3)))
        self._model.add(Dropout(dropout_par))
        self._model.add(Dense(self._num_outputs, activation='softmax'))
        self._model.compile(optimizer='Rmsprop',
                            loss='categorical_crossentropy',
                            metrics=['accuracy', top_5_acc])    
    def paper_model_MaxPooling(self,empadding_wor_len ,dropout_par , num_outputs, time_steps=500, vocab_size=20000, hidden_layer_size=128):
        """
         model Architecture:
             add GlobalMaxPooling1D

        """
        print("paper_model_MaxPooling")

        self._time_steps = time_steps
        self._vocab_size = vocab_size
        self._num_outputs = num_outputs
        self._hidden_layer_size = hidden_layer_size
        self.empadding_wor_len=empadding_wor_len
        self.dropout_par=dropout_par
        
        print("\nBuilding model...\nHidden layer size: {0}\nAnalyzing up to {1} words for each sample.".format(
            hidden_layer_size, time_steps))
        
        self._model = Sequential()
        self._model.add(Embedding(self._vocab_size, empadding_wor_len))
        self._model.add(Bidirectional(LSTM(hidden_layer_size, dropout=0.3, return_sequences=True)))
        self._model.add(GlobalMaxPooling1D())
        self._model.add(Dropout(dropout_par))
        self._model.add(Dense(self._num_outputs, activation='softmax'))
        self._model.compile(optimizer='Rmsprop',
                            loss='categorical_crossentropy',
                            metrics=['accuracy', top_5_acc])        
        
        
    def paper_model_convolution(self,empadding_wor_len ,dropout_par, num_outputs, time_steps=500, vocab_size=20000, hidden_layer_size=128):
        """
         model Architecture:
             add convolution layer 
        """
        print("paper_model_convolution")

        self._time_steps = time_steps
        self._vocab_size = vocab_size
        self._num_outputs = num_outputs
        self._hidden_layer_size = hidden_layer_size
        self.empadding_wor_len=empadding_wor_len
        self.dropout_par=dropout_par
        
        print("\nBuilding model...\nHidden layer size: {0}\nAnalyzing up to {1} words for each sample.".format(
            hidden_layer_size, time_steps))
        
        self._model = Sequential()
        self._model.add(Embedding(self._vocab_size, empadding_wor_len))
        self._model.add(Bidirectional(LSTM(hidden_layer_size, dropout=0.3, return_sequences=True)))
        self._model.add(Conv1D(hidden_layer_size,3,activation='relu'))
        self._model.add(GlobalMaxPooling1D())
        self._model.add(Dropout(dropout_par))
        self._model.add(Dense(self._num_outputs, activation='softmax'))
        self._model.compile(optimizer='Rmsprop',
                            loss='categorical_crossentropy',
                            metrics=['accuracy', top_5_acc])
    def paper_model_aggregation(self,empadding_wor_len ,dropout_par, num_outputs, time_steps=500, vocab_size=20000, hidden_layer_size=128):
        """
         model Architecture:
             add aggregation layer
        """
        print("paper_model_aggregation")

        self._time_steps = time_steps
        self._vocab_size = vocab_size
        self._num_outputs = num_outputs
        self._hidden_layer_size = hidden_layer_size
        self.empadding_wor_len=empadding_wor_len
        self.dropout_par=dropout_par
        
        print("\nBuilding model...\nHidden layer size: {0}\nAnalyzing up to {1} words for each sample.".format(
            hidden_layer_size, time_steps))
        
        self._model = Sequential()
        self._model.add(Embedding(self._vocab_size, empadding_wor_len))
        self._model.add(Bidirectional(LSTM(hidden_layer_size, dropout=0.3, return_sequences=True)))
        self._model.add(Conv1D(hidden_layer_size,3,activation='relu'))
        self._model.add(Dense(empadding_wor_len, activation='softmax'))
        self._model.add(GlobalMaxPooling1D())
        self._model.add(Dropout(dropout_par))
        self._model.add(Dense(self._num_outputs, activation='softmax'))
        self._model.compile(optimizer='Rmsprop',
                            loss='categorical_crossentropy',
                            metrics=['accuracy', top_5_acc])
        
    def full_paper_model_glove(self, empadding_wor_len,dropout_par,num_outputs, time_steps=500, vocab_size=20000, hidden_layer_size=128):
        """
         model Architecture:
             US model 
        """
        
        print("full_paper_model_glove")

        embedding_matrix=self.loadEmbeddingMatrix(empadding_wor_len,"glove")

        self._time_steps = time_steps
        self._vocab_size = vocab_size
        self._num_outputs = num_outputs
        self._hidden_layer_size = hidden_layer_size
        self.empadding_wor_len = empadding_wor_len
        self.dropout_par = dropout_par



        print("\nBuilding model...\nHidden layer size: {0}\nAnalyzing up to {1} words for each sample.".format(
            hidden_layer_size, time_steps))
        


        self._model = Sequential()
        self._model.add(Embedding(self._vocab_size,output_dim=empadding_wor_len,weights=[embedding_matrix],input_length=empadding_wor_len,trainable=False))
        self._model.add(Bidirectional(LSTM(hidden_layer_size, dropout=0.3, return_sequences=True)))
        self._model.add(Conv1D(hidden_layer_size,3,activation='relu'))
        self._model.add(Dense(empadding_wor_len))
        self._model.add(GlobalMaxPooling1D())
        self._model.add(Dropout(dropout_par))
        self._model.add(Dense(self._num_outputs, activation='softmax'))
        self._model.compile(optimizer='Rmsprop',
                            loss='categorical_crossentropy',
                            metrics=['accuracy', top_5_acc])
        
        
        
    def paper_model_connected(self,x_train,y_train,x_dev,y_dev, num_outputs, time_steps=500, vocab_size=20000, hidden_layer_size=128):
        
        print("paper_model_connected")

        self._time_steps = time_steps
        self._vocab_size = vocab_size
        self._num_outputs = num_outputs
        self._hidden_layer_size = hidden_layer_size
        self.x_train = x_train
        self.y_train = y_train
        self.x_dev=x_dev
        self.y_dev=y_dev

        """
         model Architecture:
             add aggregation layer
        """

        
        from keras.models import Model
        import tensorflow as tf

        print("\nBuilding model...\nHidden layer size: {0}\nAnalyzing up to {1} words for each sample.".format(
            128, 500))
        
        words_input = Input(shape=(None,),name='words_input')
        print(words_input.shape)
        



        emm_out=(Embedding(20000,100))(words_input)
        print(emm_out.shape)


        bi_out=(Bidirectional(LSTM(64,dropout=0.3,return_sequences=True)))(emm_out)
        conv_out=(Conv1D(128,3,padding='same',activation='relu')(emm_out))

        print(bi_out.shape)
        print(conv_out.shape)



        output = concatenate([conv_out,bi_out])
        output=(Dropout(0.4))(output)
        output=(Dense(13, activation='softmax'))(output)
        model =Model(inputs=[words_input], outputs=[output])
        model.compile(loss='sparse_categorical_crossentropy', optimizer='nadam',metrics=['accuracy'])
        
        
        print(model.summary())
        x=x_train
        x = tf.convert_to_tensor(x)      
        print(x.shape)
        y=y_train
        y= tf.convert_to_tensor(y)      
        print(y.shape)
        
        #history = model.fit(x,y, epochs=1, batch_size=512)
        
        
        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train and y_train must have the same number of samples.", x_train.shape[0],
                             y_train.shape[0])

        if x_dev.shape[0] != y_dev.shape[0]:
            raise ValueError("x_dev and y_dev must have the same number of samples.", x_dev.shape[0],
                             y_dev.shape[0])

        self._create_label_encoder(y_train)
        y_train = self._label_encoder.transform(y_train)
        self._create_label_encoder(y_dev)
        y_dev = self._label_encoder.transform(y_dev)
        print(x_train.shape)

        self._create_tokenizer(x_train)
        print("Tokenizing tweets from {0:,} users. This may take a while...".format(x_train.shape[0] + x_dev.shape[0]))
        x_dev = self._tokenize_texts(x_dev)
        x_train = self._tokenize_texts(x_train)

        y_train = keras.utils.np_utils.to_categorical(y_train, num_classes=self._num_outputs)
        y_dev = keras.utils.np_utils.to_categorical(y_dev, num_classes=self._num_outputs)
        print(y_train.shape)
        print(x_train.shape)


        if self._use_tensorboard:
            callbacks = self._generate_callbacks()
        else:
            callbacks = []
        print("Training model...")
        history = model.fit(x_train, y_train, epochs=5, batch_size=512,validation_data=(x_dev, y_dev), callbacks=callbacks)
        print(model.summary())        
        
    
        
    def train(self, x_train, y_train, x_dev, y_dev, epochs=10, reset_model=False):
        """
        Fit the model to the training data.

        :param x_train: Training samples.
        :param y_train: Training labels. Must be a vector of integer values.
        :param x_dev: Validation samples.
        :param y_dev: Validation labels. Must be a vector of integer values.
        :param epochs: Number of times to train on the whole data set. Default: 7
        :param reset_model: If this is set to True, it will discard any previously trained model and start from scratch.
        :return:
        :raises: ValueError: If the number of training samples and the number of labels do not match.
        """

        if x_train.shape[0] != y_train.shape[0]:
            raise ValueError("x_train and y_train must have the same number of samples.", x_train.shape[0],
                             y_train.shape[0])

        if x_dev.shape[0] != y_dev.shape[0]:
            raise ValueError("x_dev and y_dev must have the same number of samples.", x_dev.shape[0],
                             y_dev.shape[0])

        #self._create_label_encoder(y_train)
       # y_train = self._label_encoder.transform(y_train)
        #self._label_encoder=None
        #self._create_label_encoder(y_dev)
        #y_dev = self._label_encoder.transform(y_dev)
        #self._label_encoder=None
        if self._label_encoder is None: self._create_label_encoder(y_train)
        y_train = self._label_encoder.transform(y_train)
        self._create_label_encoder(y_dev)
        y_dev = self._label_encoder.transform(y_dev)

        if self._tokenizer is None: self._create_tokenizer(x_train)
        print("Tokenizing tweets from {0:,} users. This may take a while...".format(x_train.shape[0] + x_dev.shape[0]))
        x_dev = self._tokenize_texts(x_dev)
        x_train = self._tokenize_texts(x_train)

        y_train = keras.utils.np_utils.to_categorical(y_train, num_classes=self._num_outputs)
        y_dev = keras.utils.np_utils.to_categorical(y_dev, num_classes=self._num_outputs)

        if self._use_tensorboard:
            callbacks = self._generate_callbacks()
        else:
            callbacks = []
        print("Training model...")
        history = self._model.fit(x_train, y_train, epochs=epochs, batch_size=self._batch_size,
                                  validation_data=(x_dev, y_dev), callbacks=callbacks)
        print(self._model.summary())
        return history

    def predict(self, x):
        """
        Predict the location of the given samples.

        :param x: A vector of tweets. Each row corresponds to a single user.
        :return: The prediction results.
        """
        x = self._tokenize_texts(x)
        predictions = self._model.predict(x, batch_size=self._batch_size)

        predictions = np.argmax(predictions, axis=1)
        #predictions = self._label_encoder.inverse_transform(predictions)
        return predictions

    def evaluate(self, x_test, y_test):
        """
        Get the loss, accuracy and top 5 accuracy of the model.

        :param x_test: Evaluation samples.
        :param y_test: Evaluation labels.
        :return: A dictionary of metric, value pairs.
        """

        x_test = self._tokenize_texts(x_test)

        y_test = self._label_encoder.transform(y_test)
        y_test = keras.utils.np_utils.to_categorical(y_test, num_classes=self._num_outputs)
        metrics = self._model.evaluate(x_test, y_test, batch_size=self._batch_size)
        d = {}
        for i in range(len(self._model.metrics_names)):
            d[self._model.metrics_names[i]] = metrics[i]
        return d

    def load_saved_model(self, filename):
        """
        Load a previously trained model from disk.

        :param filename: The H5 model.
        :return:
        """
        model_filename = filename + ".h5"
        tokenizer_filename = filename + ".tokenizer"
        label_encoder_filename = filename + ".labelencoder"
        metadata_filename = filename + ".meta"

        if path.exists(model_filename):
            print("Loading saved model...")
        else:
            raise Exception("Saved model {0} does not exist.".format(model_filename), model_filename)

        self._model = keras.models.load_model(model_filename, custom_objects={'top_5_acc': top_5_acc})

        with open(tokenizer_filename, 'rb') as handle:
            self._tokenizer = pickle.load(handle)

        with open(label_encoder_filename, 'rb') as handle:
            self._label_encoder = pickle.load(handle)

        with open(metadata_filename, 'rb') as handle:
            metadata = pickle.load(handle)
            self._vocab_size = metadata['vocab_size']
            self._hidden_layer_size = metadata['hidden_layer_size']
            self._num_outputs = metadata['num_outputs']
            self._time_steps = metadata['time_steps']

    def save_model(self, filename):
        """
        Save the current model and trained weights for later use.

        :param filename: Prefix for the model filenames.
        """

        model_filename = filename + ".h5"
        tokenizer_filename = filename + ".tokenizer"
        label_encoder_filename = filename + ".labelencoder"
        metadata_filename = filename + ".meta"

        self._model.save(model_filename)

        with open(tokenizer_filename, 'wb') as handle:
            pickle.dump(self._tokenizer, handle)

        with open(label_encoder_filename, 'wb') as handle:
            pickle.dump(self._label_encoder, handle)

        metadata = {'hidden_layer_size': self._hidden_layer_size,
                    'vocab_size': self._vocab_size,
                    'num_outputs': self._num_outputs,
                    'time_steps': self._time_steps}

        with open(metadata_filename, 'wb') as handle:
            pickle.dump(metadata, handle)

    def _generate_callbacks(self):
        now = time.time()
        log_dir = './.tensorboard_dir/{0}'.format(str(int(now)))
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0,
                                                           write_graph=True, batch_size=self._batch_size,
                                                           write_images=True)
        return [tensorboard_callback]

    def _create_tokenizer(self, x_train):
        print("Building tweet Tokenizer using a {0:,} word vocabulary. This may take a while...".format(
            self._vocab_size))
        self._tokenizer = Tokenizer(num_words=self._vocab_size, lower=True,
                                    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{}~\t\n')
        self._tokenizer.fit_on_texts(x_train)

    def _create_label_encoder(self, y_train):
        self._label_encoder = preprocessing.LabelEncoder()
        self._label_encoder.fit(y_train)

    def _tokenize_texts(self, texts):
        texts = self._tokenizer.texts_to_sequences(texts)
        texts = pad_sequences(texts, maxlen=self._time_steps, truncating='pre')
        return texts
