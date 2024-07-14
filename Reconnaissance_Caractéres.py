# https://www.kaggle.com/datasets/crawford/emnist (lien de telechargement du dataset)

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd  # Bibliotheque de lecture de tableaux (csv)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
file_csv = os.path.join(BASE_DIR, "emnist-balanced-train.csv")

print("Chargement des données EMNIST Balanced Train")
train_data = pd.read_csv(file_csv)

print("Filtre pour inclure uniquement les lettres majuscules dans l'ensemble de train")

train_letters_mask = (train_data.iloc[:, 0] >= 10) & (train_data.iloc[:, 0] <= 35) | (train_data.iloc[:, 0] < 10)
train_data = train_data.loc[train_letters_mask]



print("Chargement des données EMNIST Balanced Test")
file_csv = os.path.join(BASE_DIR, "emnist-balanced-test.csv")
test_data = pd.read_csv(file_csv)



print("Filtre pour inclure uniquement les lettres majuscules dans l'ensemble de test")
test_letters_mask = (test_data.iloc[:, 0] >= 10) & (test_data.iloc[:, 0] <= 35) | (test_data.iloc[:, 0] < 10)
test_data = test_data.loc[test_letters_mask]


print("Séparation des caractéristiques (pixels) et des étiquettes (lettres ou chiffres) pour l'ensemble de train")

X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values

print("Séparation des caractéristiques (pixels) et des étiquettes (lettres ou chiffres) pour l'ensemble de test")

X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

print("Création du classifieur k-nn")

knn = KNeighborsClassifier(n_neighbors=8)

print("Entraînement du classifieur")

knn.fit(X_train, y_train)

print("Évaluation des performances du modèle sur l'ensemble de test")

accuracy = knn.score(X_test, y_test)

print("Exactitude du modèle : ", accuracy)

# Fonction pour prédire une lettre ou un chiffre à partir d'une image

def predict_symbol(image):
    image = np.array(image).reshape(1, -1)
    predicted_symbol = knn.predict(image)
    print(predicted_symbol)

    if predicted_symbol[0] < 10:
        # Renvoyer un chiffre en tant que chaîne de caractères
        return str(predicted_symbol[0])

    else:
        # Renvoyer une lettre majuscule en tant que caractère
        return chr(predicted_symbol[0] + 55)

# Tranformation d'une image en list utilisable dans predict_symbole

def image_normalized(image_path):
    # Convertir l'image en niveaux de gris
    image = Image.open(image_path).convert("L")

    # Inversion du noir et blanc (pour utilisation dans la fct)
    image = image.point(lambda pixel: 255 - pixel)

    # Redimensionner l'image à la taille attendue (28 pixels max)
    h, w = image.size
    print("taille initiale", (h, w))
    maxi = max(h, w)
    im = None
    if abs(h-w) < (3/4)*maxi:
        h, w = int(28 * h / maxi), int(28 * w / maxi)
        print(h,w)
        print("taille max 28", (h, w))
        resized_image = image.resize((h, w))

        # Vignette centrée sur carré 28x28 pixels
        im = Image.new("RGB", (28, 28))
        im.paste(resized_image, box=((28 - h) // 2, (28 - w) // 2))

        # Orientation pour analyse knn

        im = im.rotate(90).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return im


def img_to_list(img):
    arr = np.array(img.getdata())
    pixel_values = list(arr[:, 0])

    return pixel_values

print("Exemple d'utilisation")

# Récupérer la liste des fichiers dans le dossier "lettres"
lettres_folder = os.path.join(BASE_DIR, "lettres")
image_files = os.listdir(lettres_folder)

for file_name in image_files:
    file_png = os.path.join(lettres_folder, file_name)
    image_to_predict = image_normalized(file_png)
    if image_to_predict != None:
        pixel_values = img_to_list(image_to_predict)
        predicted_symbol = predict_symbol(pixel_values)
        print("Fichier :", file_name)
        print("Symbole prédit :", predicted_symbol)


