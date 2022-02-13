from pdf2image import convert_from_path
import os

def creation_images(dossier_traitement, dossier_destination):

    docs = os.listdir(dossier_traitement)
    name = "Document"
    compteur = 0

    for doc in docs:
        converted_doc = convert_from_path(os.path.join(dossier_traitement, doc))
        converted_doc[0].save(dossier_destination + "/" + name + " " + str(compteur) + ".jpeg", "JPEG")
        compteur+=1

creation_images("../data/pdf/pdf with zoi", 'train/')