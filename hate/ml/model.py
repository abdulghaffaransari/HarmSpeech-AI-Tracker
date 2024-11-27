# Creating model architecture.
from hate.entity.config_entity import ModelTrainerConfig
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM,Activation,Dense,Dropout,Input,Embedding,SpatialDropout1D
from hate.constants import *
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Bidirectional

class ModelArchitecture:

    def __init__(self):
        pass

    
    def get_model(self):
        
        
        model = Sequential()
        model.add(Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN))
        model.add(SpatialDropout1D(0.3))
        model.add(Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3)))
        model.add(Dense(1, activation=ACTIVATION))
        model.build(input_shape=(None, MAX_LEN))
        model.summary()
        # Compile the model
        model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=METRICS)
        # Add EarlyStopping to prevent overfitting
        early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)
        
        return model