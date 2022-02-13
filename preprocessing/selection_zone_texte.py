import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def decomposition_img(lien_img):
    img = cv.imread(lien_img)

    #Phase de transformation
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_blur = cv.blur(img_gray, (33,33), 0)
    ret, thresh = cv.threshold(img_blur, 220, 255, cv.THRESH_BINARY)
    img_canny = cv.Canny(thresh,125,175)
    dilated = cv.dilate(img_canny, (15,15), iterations=65)
    dilated_blur = cv.blur(dilated, (15,15))

    #Détection des contours
    contours, hierarchies = cv.findContours(dilated_blur, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)

    list_img = []
    for idx in range(len(contours[:10])):

        mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
        cv.drawContours(mask, contours, idx, 255, -1) # Draw filled contour in mask
        out = np.zeros_like(img) # Extract out the object and place into output image
        out[mask == 255] = img[mask == 255]

        # Now crop
        y, x, _ = np.where(mask == 255)
        (topy, topx) = (np.min(y), np.min(x))
        (bottomy, bottomx) = (np.max(y), np.max(x))

        if(bottomy - topy > 30) and (bottomx - topx > 30):

            blank = np.zeros(img.shape[:2], dtype='uint8')
            rectangle = cv.rectangle(blank.copy(), (topx, topy), (bottomx, bottomy), 255, -1)
            out = cv.bitwise_and(img,img, mask=rectangle)
            out = out[topy:bottomy+1, topx:bottomx+1]
            img_gray_out = cv.cvtColor(out, cv.COLOR_BGR2GRAY)
            list_img.append(img_gray_out)
    
    return list_img

lili_res = decomposition_img('train/Document 18.jpeg')