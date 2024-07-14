import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# Chargement des données EMNIST Balanced Train
train_data = pd.read_csv('emnist-balanced-train.csv')

# Filtre pour inclure uniquement les lettres majuscules dans l'ensemble de train
train_letters_mask = (train_data.iloc[:, 0] >= 10) & (train_data.iloc[:, 0] <= 35) | (train_data.iloc[:, 0] < 10)
train_data = train_data.loc[train_letters_mask]

# Chargement des données EMNIST Balanced Test
test_data = pd.read_csv('emnist-balanced-test.csv')

# Filtre pour inclure uniquement les lettres majuscules dans l'ensemble de test
test_letters_mask = (test_data.iloc[:, 0] >= 10) & (test_data.iloc[:, 0] <= 35) | (test_data.iloc[:, 0] < 10)
test_data = test_data.loc[test_letters_mask]

# Séparation des caractéristiques (pixels) et des étiquettes (lettres ou chiffres) pour l'ensemble de train
X_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values

# Séparation des caractéristiques (pixels) et des étiquettes (lettres ou chiffres) pour l'ensemble de test
X_test = test_data.iloc[:, 1:].values
y_test = test_data.iloc[:, 0].values

# Listes pour stocker les valeurs de K et les précisions correspondantes
k_values = []
accuracies = []

# Boucle sur les valeurs de K de 1 à 100
for k in range(1, 101):
    # Création du classifieur k-nn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Entraînement du classifieur
    knn.fit(X_train, y_train)

    # Évaluation des performances du modèle sur l'ensemble de test
    accuracy = knn.score(X_test, y_test)

    # Ajout des valeurs de K et de la précision à leurs listes respectives
    k_values.append(k)
    accuracies.append(accuracy)

    print("K =", k, "- Exactitude du modèle :", accuracy)

# Tracé de la précision en fonction de K
plt.plot(k_values, accuracies)
plt.xlabel('Valeur de K')
plt.ylabel('Précision')
plt.title('Précision du modèle en fonction de K')
plt.show()
