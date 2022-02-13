import numpy as np
import cv2 as cv
import os
from pdf2image import convert_from_path
import pytesseract
import re
from PIL import Image

def creation_images(dossier_destination, dossier_traitement):

    name = "mon_pdf"
    converted_doc = convert_from_path(os.path.join(dossier_traitement))
    converted_doc[0].save(dossier_destination + "/" + name + ".jpeg", "JPEG")


def decomposition_img(lien_img):
    img = cv.imread(lien_img)

    #Phase de transformation
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.blur(img_gray, (33,33), 0)
    ret, thresh = cv.threshold(img_blur, 220, 255, cv.THRESH_BINARY)
    img_canny = cv.Canny(thresh,125,175)
    dilated = cv.dilate(img_canny, (15,15), iterations=65)
    dilated_blur = cv.blur(dilated, (15,15))

    #Détection des contours en prenant les plus importants quand y en a trop
    contours, hierarchies = cv.findContours(dilated_blur, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    if (len(contours)> 25):
        contours = sorted(contours, key=cv.contourArea, reverse=True)[:25]

    #Sauvegarde pour présentation
    img_contour = cv.drawContours(img.copy(), contours, -1, (0,255,0), 3)
    cv.imwrite('pdf/blur_1.jpeg', img_blur)
    cv.imwrite('pdf/canny.jpeg', dilated)
    cv.imwrite('pdf/contour.jpeg', img_contour)

    list_img = []

    for idx in range(len(contours)):

        mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
        cv.drawContours(mask, contours, idx, 255, -1) # Draw filled contour in mask
        out = np.zeros_like(img) # Extract out the object and place into output image
        out[mask == 255] = img[mask == 255]

        # Now crop
        y, x, _ = np.where(mask == 255)
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))

        if(bottomy - topy > 30) and (bottomx - topx > 30):

            out = out[topy:bottomy+1, topx:bottomx+1]
            img_gray_out = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
            list_img.append(img_gray_out)
    
    return list_img


def sauvegarde_decoupe(list_img, dossier_destination):
    compteur = 0
    imgs = os.listdir(dossier_destination)
    for im in imgs:
        os.remove(dossier_destination + "/" + im)
    for img in list_img:
        cv.imwrite(dossier_destination + "/" + str(compteur) + ".jpeg", img)
        compteur+=1


def image_to_csv_tesseract(dossier_traitement):

    docs = os.listdir(dossier_traitement)
    texte_extract = ""
    for doc in docs:
        text = pytesseract.image_to_string(Image.open(os.path.join(dossier_traitement, doc)))
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r";+", "", text)
        texte_extract+=text
    
    return texte_extract