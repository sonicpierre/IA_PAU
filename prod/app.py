import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from util_class import RC_model, Reservoir, tensorPCA
from extraction_texte import creation_images, decomposition_img, sauvegarde_decoupe, image_to_csv_tesseract
from creation_equipe import creation_description_equipe
from classification_model import construction_prediction, pred_doc_complet
from gestion_BDD import download_blob, list_blobs
from preprocessing_mod import text_preprocessing, text_to_model, text_preprocessing_network
from pdf2image import convert_from_path
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/virgaux/IA_PAU_2022/iapau-3afbf-firebase-adminsdk-oag2r-2368a91e22.json"
texte = ""

col1, col2 = st.columns([2, 4])

col1.image('assets/logos/cap_logo.png', width = 200)
col1.image('assets/logos/logo_ia_pau.png', width = 200)
col2.title("IA Pau 4 : Prediction sous pression")

with open("texte/description_asso_ia_pau.txt", 'r') as f:
    description_asso = f.readlines()

col2.text("".join(description_asso))

st.markdown('## Equipe')

creation_description_equipe()

st.markdown('## Sujet')

with open("texte/description_challenge_part1.txt", 'r') as f:
    description_sujet_1 = f.readlines()

st.text("".join(description_sujet_1))

st.video("https://www.youtube.com/watch?v=TIlp4_9ZlZc")

with open("texte/description_challenge_part2.txt", 'r') as f:
    description_sujet_2 = f.readlines()

st.text("".join(description_sujet_2))

st.markdown('## Notre étude')

st.write("Il est possible ici de choisir un fichier PDF pour faire le traitement de façon intéractive.")

col1, col2 = st.columns([2, 2])
with col2:
    option = st.selectbox("Depuis Google Cloud Plateform", ['Document'] + list_blobs('iapau-3afbf.appspot.com'))
    
    if option and option != 'Document' and not os.path.exists("assets/Doc_complet.pdf"):
        with st.spinner('Wait for it...'):
            download_blob('iapau-3afbf.appspot.com', option, 'pdf/mon_pdf.pdf')
            creation_images("pdf","pdf/mon_pdf.pdf")
            liste_img = decomposition_img('pdf/mon_pdf.jpeg')
            sauvegarde_decoupe(liste_img, "pdf/image_decoupe")
            texte = image_to_csv_tesseract("pdf/image_decoupe")
            with open("pdf/mon_pdf.txt", 'w') as f:
                f.write(texte)
        option = 'Document'

with col1:
    uploaded_file = st.file_uploader("Depuis votre ordinateur")
    
    if uploaded_file and option =='Document' and not os.path.exists("assets/Doc_complet.pdf"):
        with st.spinner('Wait for it...'):
            bytes_data = uploaded_file.read()
            with open("pdf/mon_pdf.pdf", 'wb') as f:
                description_asso = f.write(bytes_data)

            creation_images("pdf","pdf/mon_pdf.pdf")
            liste_img = decomposition_img('pdf/mon_pdf.jpeg')
            sauvegarde_decoupe(liste_img, "pdf/image_decoupe")
            texte = image_to_csv_tesseract("pdf/image_decoupe")
            with open("pdf/mon_pdf.txt", 'w') as f:
                f.write(texte)


st.markdown('## Optical Character Recognition')

st.write("Il a fallu tout d'abord preprocessé les PDF pour permettre à l'OCR Tesseract de mieux reconnaître les charactères présents sur le document.")

col1, col2, col3, col4 = st.columns([2, 2, 2, 2])

col1.image('pdf/mon_pdf.jpeg')
col2.image('pdf/blur_1.jpeg')
col3.image('pdf/canny.jpeg')
col4.image('pdf/contour.jpeg')

st.write("Grâce à Tesseract on peut ainsi faire ressortir les mots associés au document.")

if(texte == ""):
    with open("pdf/mon_pdf.txt") as f:
        texte = " ".join(f.readlines())

wordcloud = WordCloud().generate(texte)       
fig = plt.figure(figsize = (10, 10), num = 1, clear = True)
ax =plt.subplot(1, 1, 1, xticks = [], yticks = [], frameon = False)
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis = False
st.pyplot(fig)

st.markdown('## Classification unique')
st.write("On vectorise au final nos mots ou nos phrases et on applique les modèles de prédiction.")

vec = text_preprocessing(texte)
vec2 = text_to_model(texte)
vec3 = text_preprocessing_network(texte)
construction_prediction(vec, vec2, vec3)

st.markdown('## Document entier')

st.write("Traiter des documents entier d'un seul coup pour mettre de ccibler directement les pages qui nous interessent !")

uploaded_file_complet = st.file_uploader("Document complet")

if uploaded_file_complet:
    with st.spinner('Wait for it...'):
        bytes_data = uploaded_file_complet.read()
        with open("assets/Doc_complet.pdf", 'wb') as f:
            description_asso = f.write(bytes_data)

        images = convert_from_path('assets/Doc_complet.pdf')
        compteur = 0
        for im in images:
            im.save("pdf_complet/" + str(compteur) + ".jpeg", format="jpeg")
            compteur += 1
        
        dico = pred_doc_complet("pdf_complet")
        liste_col = []
        for i in range(len(dico)):
            liste_col.append(2)
        pan = st.columns(liste_col)

        compteur = 0
        liste_interet = []
        for i in dico:
            pan[compteur].image("pdf_complet/" + i)
            if(dico[i] == 1):
                pan[compteur].markdown("<font color='green'>Page ZOI</font>", unsafe_allow_html = True)
                liste_interet.append(str(compteur + 1))
            else:
                pan[compteur].markdown("<font color='red'>Pas ZOI</font>", unsafe_allow_html = True)

            compteur+=1
        
        if len(liste_interet) != 0:
            st.markdown("## Vous pouvez aller voir la page " + " ".join(liste_interet))
    
    os.remove("assets/Doc_complet.pdf")
    for im in os.listdir('pdf_complet/'):
        os.remove('pdf_complet/' + im)