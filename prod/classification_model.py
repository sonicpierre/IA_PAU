import streamlit as st
import os
import re
import pytesseract
import joblib
import pickle
from tensorflow.keras.models import load_model
from preprocessing_mod import text_preprocessing

loaded_logistic = pickle.load(open('model/LogisticRegressionClassifier2.sav', 'rb'))
loaded_reservoir_model = pickle.load(open("model/classifier_mlp", "rb" ))
loaded_lstm = load_model("model/BiLSTM.h5")

def construction_prediction(vec, vec2, vec3):
    col_1, col_2, col_3 = st.columns([2,2,2])

    pred = loaded_logistic.predict(vec)[0]
    proba = round(max(loaded_logistic.predict_proba(vec)[0]),4)

    col_1.markdown("Logistic", unsafe_allow_html=True)

    if(pred == 1):

        col_1.image('assets/graphics/pouce_leve.png')
        col_1.write("Il s'agit d'une page ZOI !")
        col_1.write("Confiance : " + str(proba))

    else:
        col_1.image('assets/graphics/pouce_baisse.png')
        col_1.write("Il ne s'agit pas d'une page ZOI !")
        col_1.write("Confiance : " + str(proba))

    pred2 = loaded_lstm.predict(vec3)
    pred2 = pred2.sum()
    col_2.markdown("Réseau LSTM", unsafe_allow_html=True)

    if(pred2 == 1):
        col_2.image('assets/graphics/pouce_leve.png')
        col_2.write("Il s'agit d'une page ZOI !")

    else:
        col_2.image('assets/graphics/pouce_baisse.png')
        col_2.write("Il ne s'agit pas d'une page ZOI !")

    pred3 = loaded_reservoir_model.predict(vec2)
    proba_3 = round(loaded_reservoir_model.predict_proba(vec2),4)
    col_3.markdown("Modèle réservoir", unsafe_allow_html=True)

    if(pred3 == 1):

        col_3.image('assets/graphics/pouce_leve.png')
        col_3.write("Il s'agit d'une page ZOI !")
        col_3.write("Confiance : " + str(proba_3))
    else:
        col_3.image('assets/graphics/pouce_baisse.png')
        col_3.write("Il ne s'agit pas d'une page ZOI !")
        col_3.write("Confiance : " + str(proba_3))


def pred_doc_complet(dossier):
    predictions = {}
    for im in os.listdir(dossier):
        text = pytesseract.image_to_string(dossier + "/" + im)
        text = re.sub(r";+", " ", text)
        text = re.sub(r"\s+", " ", text)
        vec = text_preprocessing(text)
        predictions[im] = loaded_logistic.predict(vec)[0]
    return predictions