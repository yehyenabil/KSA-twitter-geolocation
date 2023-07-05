# KSA-twitter-geolocation
=============================
Twitter Geolocation Predictor
This is a deep-learning tool to predict the location of a Twitter user
based solely on the text content of his/her tweets without any other
form of metadata.


Overview
--------

The Twitter Geolocation Predictor is a Recurrent Neural Network
classifier. Every training sample is a collection of tweets labeled with
a location (e.g. country, state, city, etc.). The model will
tokenize all tweets into a sequence of words, and feed them into an
`Embedding Layer <https://en.wikipedia.org/wiki/Word_embedding>`__. The
embeddings will learn the meaning of words and use them as input for two
stacked `Long-Short Term
Memory <http://colah.github.io/posts/2015-08-Understanding-LSTMs/>`__
layers. A `Softmax <https://en.wikipedia.org/wiki/Softmax_function>`__
fully-connected layer at the end yields the classification result.

etting Started
---------------

Dependencies
~~~~~~~~~~~~
1. Python 3.5
2. tensorflow
3. keras
4. nltk
5. pandas
6. numpy
7. sqlalchemy
8. sklearn
9. psycopg2


Installation
~~~~~~~~~~~~

Clone the repository and install all the dependencies using pip.

.. code:: console

    $ git clone git@github.com:yehyenabil/KSA-twitter-geolocation.git
    $ cd KSA-twitter-geolocation
    $ sudo pip3 install -r requirements.txt

This will install the latest CPU version of Tensorflow. If you would
like to run on a GPU, follow the Tensorflow-GPU `installation
instructions <https://www.tensorflow.org/install/>`__.
Pre-Processing your own data
----------------------------

+------------------------------------------------------------------+------------+
| Tweet Text                                                       | Location   |
+==================================================================+============+
| Hello world! This is a tweet. <eot> This is another tweet. <eot> | Florida    |
+------------------------------------------------------------------+------------+
| Going to see Star Wars tonite!                                   | Puerto Rico|
+------------------------------------------------------------------+------------+
| Pizza was delicious! <eot> I'm another tweeeeeet <eot>           | California |
+------------------------------------------------------------------+------------+


Given a raw dataset stored in a CSV file like the one shown above, we can preprocess said data using :code:`twgeo.data.input.read_csv_data()`. This function will:

    1. Tokenize the tweet text.
    2. Limit repeated characters to a maximum of 2. For example: 'Greeeeeetings' becomes 'Greetings'.
    3. Perform `Porter stemming  <https://en.wikipedia.org/wiki/Stemming>`_ on each token.
    4. Convert each token to lower case.

The location data may be any string or integer value.

.. code:: python

    import twgeo.data.input as input
    tweets, locations = input.read_csv_data('mydata.csv', tweet_txt_column_idx=0, location_column_idx=1)


Training the Model
------------------

.. code:: python

    from twgeo.models.geomodel import Model
    from twgeo.data import twus
    
    # x_train is an array of text. Each element contains all the tweets for a given user. 
    # y_train is an array of integer values, corresponding to each particular location we want to train against.
    x_train, y_train, x_dev, y_dev, x_test, y_test = twus.load_state_data(size='small')

    # num_outputs is the total number of possible classes (locations). In this example, 50 US states plus 3 territories.
    # time_steps is the total number of individual words to consider for each user.
    # Some users have more tweets then others. In this example, we are capping it at a total of 500 words per user.
    geoModel = Model(batch_size=64)
    geoModel.build_model(num_outputs=53, time_steps=500,vocab_size=20000)
                     
    geoModel.train(x_train, y_train, x_dev, y_dev, epochs=5)
    geoModel.save_model('mymodel')

Making Predictions
------------------

.. code:: ipython

    In [1]: from twgeo.models.geomodel import Model
    Using TensorFlow backend.

    In [2]: from twgeo.data import twus_dataset as twus

    In [3]: x_train, y_train, x_dev, y_dev, x_test, y_test = twus.load_state_data(size='small')

    In [4]: geoModel = Model()

    In [5]: geoModel.load_saved_model('mymodel')
    Loading saved model...

    In [6]: geoModel.predict(x_test)
    Out[6]: array(['CA', 'FL', 'NY', ..., 'TX', 'MA', 'KY'], dtype=object)


