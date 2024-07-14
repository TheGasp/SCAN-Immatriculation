import numpy as np
import cv2
import os

def seuillage(image, seuil):
    # Parcourir tous les pixels de l'image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Appliquer le seuillage
            if image[i, j] < seuil:
                image[i, j] = 0  # Noir
            else:
                image[i, j] = 255  # Blanc
    return image

#Inistialiation de la plaque
image_plaque = cv2.imread('Plaque5.png', 0)
height, width = image_plaque.shape[:2]
transformed = seuillage(image_plaque, 100)
transformed = cv2.bitwise_not(image_plaque)  # Inversion du noir et blanc (pour utilisation dans la fonction)

#Decoupage de la plaque
def decoup_plaque (image):
    detect = False
    x_start,x_end = None,None
    L = []
    for i in range(width):
        n_line = 0
        for j in range(height):
            if image[j][i] == 255:
                if detect == False: #Est ce qu'un chiffre est deja en train d'etre detecté
                    detect = True
                    x_start = i
                    x_end = i

                else :
                    if i < x_start :
                        x_start = i
                    if i > x_end :
                        x_end = i

            elif detect == True:
                n_line += 1

        if n_line == height: #Fin de collone
            if x_start != None:
                detect = False
                if x_start != x_end: #Pour eviter les pixels seuls
                    L.append([x_start,x_end])

    return L

def nettoyage():
    folder_path = "lettres"
    # Parcourir tous les fichiers et dossiers dans le dossier
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        os.remove(file_path)


def crop_image(image):
    L = decoup_plaque(image)
    nettoyage()
    for i in range(len(L)):
        lettre = L[i]
        cropped_image = image_plaque[0:height, lettre[0]:lettre[1]].copy()

        #Enregistrement
        letter_path = os.path.join("lettres", f"{i}.png")
        cv2.imwrite(letter_path, cropped_image)


crop_image(transformed)




