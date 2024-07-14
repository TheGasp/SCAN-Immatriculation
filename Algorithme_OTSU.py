import numpy as np
import cv2
import matplotlib.pyplot as plt

def find_canny_threshold(image):
    # Conversion de l'image en niveaux de gris
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calcul de l'histogramme normalisé
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

    hist_norm = hist.ravel() / hist.sum() # Méthode ravel applatit l'histogramme (tableau à 1 dimension)

    # Calcul de la somme cumulée normalisée
    cumsum = np.cumsum(hist_norm)

    # Calcul de la somme cumulée moyenne
    cumsum_mean = np.cumsum(hist_norm * np.arange(256))

    threshold = 0
    var_max = 0
    sigma = []

    # Recherche du seuil optimal
    for t in range(256):
        p1 = cumsum[t] # Proba que le pixel t
        p2 = 1 - p1

        # Si un des pixels voisins vaut 0 (noir)
        if p1 == 0 or p2 == 0: # On saute l'itération si la condition est vérifiée
            continue

        mean1 = cumsum_mean[t] / p1
        mean2 = (cumsum_mean[-1] - cumsum_mean[t]) / p2

        var = p1 * p2 * (mean1 - mean2)**2 # Formule de l'algo de Otsu
        sigma.append(var)

        if var > var_max:
            var_max = var
            threshold = t

    print(threshold)

    # Configuration de la figure avec deux axes Y
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax2 = ax1.twinx()

    # Tracé de l'histogramme normalisé
    ax1.plot(hist_norm, color='black', label='Histogramme')
    ax1.set_xlabel('Niveau de gris')
    ax1.set_ylabel('Moyenne')
    ax1.set_title('Histogramme normalisé et Variance en fonction du niveau de gris')

    # Tracé de la variance
    ax2.plot(sigma, color='red', label='Variance')
    ax2.set_ylabel('Variance(σ^2)')

    # Positionnement de la légende en haut à droite
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc='upper right', bbox_to_anchor=(1.0, 1.0))

    plt.tight_layout()
    plt.show()

# Charger l'image
image = cv2.imread('voiture4.png')
find_canny_threshold(image)


