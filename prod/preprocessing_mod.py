from scipy import sparse
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.wordnet import WordNetLemmatizer
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def text_preprocessing(pdf_text):
    stop_words = set(stopwords.words("english"))
    # Suppression des ponctuations
    text = re.sub('[^a-zA-Z]', ' ', pdf_text)
    # convertir en minuscule
    text = text.lower()
    # Suppression des balises
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    # suppression des caracères spéciaux
    text=re.sub("(\\d|\\W)+"," ",text)
    # Conversion en liste
    text = text.split()
    # normalisation de type lammélisation (on extrait de chaque mot les expressions non régulières)
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in  
            stop_words] 
    text = " ".join(text)
    cv = CountVectorizer(stop_words = stop_words, max_features = 50, ngram_range = (1,3))
    X = cv.fit_transform([text])
    X1 = sparse.csr_matrix.toarray(X)
    return(X1)


def text_preprocessing_network(pdf_text):
        max_len = 50 
        trunc_type = "post" 
        padding_type = "post" 
        oov_tok = "<OOV>" 
        vocab_size = 500
        tokenizer = Tokenizer(num_words = vocab_size, char_level=False, oov_token = oov_tok)
        tokenizer.fit_on_texts(pdf_text)
        word_index = tokenizer.word_index
        tot_words = len(word_index)
        # After Tokenization we will Sequence and padd on training and testing 
        training_sequences = tokenizer.texts_to_sequences(pdf_text)
        training_padded = pad_sequences (training_sequences, maxlen = max_len, padding = padding_type, truncating = trunc_type )
        return training_padded

def text_to_model(pdf_text):
    """ Extraction des données des fiucher pdf"""

    #   Récupération de ma liste des stop words
    stop_words = set(stopwords.words("english"))

    # Suppression des ponctuations
    text = re.sub('[^a-zA-Z]', ' ', pdf_text)
    
    # convertir en minuscule
    text = text.lower()
    
    # Suppression des balises
    text=re.sub("&lt;/?.*?&gt;"," &lt;&gt; ",text)
    
    # suppression des caracères spéciaux
    text=re.sub("(\\d|\\W)+"," ",text)
    
    # Conversion en liste
    text = text.split()
    
    # Lémmatisation
    lem = WordNetLemmatizer()
    text = [lem.lemmatize(word) for word in text if not word in  
            stop_words] 
    text = " ".join(text)

    #   Conversion du texte en vecteurs
    #   Récupérer CV depuis un fichier
    
    cv = pickle.load( open("model/CountVectorizer", "rb" ) )
    X = cv.transform([text])

    #   Conversion de la matrice en entier
    X = sparse.csr_matrix.toarray(X)
    X = X.reshape((1, X.shape[1], 1))

    return X