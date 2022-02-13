import streamlit as st

def creation_description_equipe():
    with open("assets/equipe/Samuel_desc.txt", 'r') as f:
        description_samuel = f.readlines()

    col_1, col_2 = st.columns([4, 2])
    col_2.image('assets/equipe/Samuel.png')
    col_1.text("".join(description_samuel))

    with open("assets/equipe/Carlos_desc.txt", 'r') as f:
        description_carlos = f.readlines()

    col_1, col_2 = st.columns([2, 4])
    col_1.image('assets/equipe/Carlos.png')
    col_2.text("".join(description_carlos))

    with open("assets/equipe/Pierre_desc.txt", 'r') as f:
        description_pierre = f.readlines()

    col_1, col_2 = st.columns([4, 2])
    col_2.image('assets/equipe/Pierre.png')
    col_1.text("".join(description_pierre))

    with open("assets/equipe/Wael_desc.txt", 'r') as f:
        description_wael = f.readlines()

    col_1, col_2 = st.columns([2, 4])
    col_1.image('assets/equipe/Wael.png')
    col_2.text("".join(description_wael))

