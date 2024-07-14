import numpy as np
import cv2

def convolve(image, kernel):
    height, width = image.shape
    k_height, k_width = kernel.shape
    padding = k_width // 2  # Calcul du padding pour maintenir la taille de l'image

    # Création d'une copie de l'image avec un padding
    padded_image = np.pad(image, padding, mode='constant')

    # Création de l'image résultante de la convolution
    convolved_image = np.zeros_like(image)

    # Parcours de chaque pixel de l'image
    for i in range(height):
        for j in range(width):
            # Extraction de la région d'intérêt autour du pixel
            region = padded_image[i:i + k_height, j:j + k_width]

            # Application du noyau en effectuant une multiplication élément par élément
            result = np.sum(region * kernel)

            # Stockage du résultat dans l'image convoluée
            convolved_image[i, j] = result

    return convolved_image

def grayscale(image):
    # Conversion de l'image en niveaux de gris
    return np.dot(image[...,:3], [0.2989, 0.5870, 0.1140])
    #Formule cacul pixel gris (en lien avec la luminosité visible d'une couleur) : pixel_gris = (pixel_rouge * 0.2989) + (pixel_vert * 0.5870) + (pixel_bleu * 0.1140)

def gaussian_blur(image, kernel_size=5, sigma=1.4):
    # Création du noyau gaussien
    kernel = np.fromfunction(lambda x, y: (1/(2*np.pi*sigma**2)) * np.exp(-((x-(kernel_size-1)//2)**2 + (y-(kernel_size-1)//2)**2)/(2*sigma**2)), (kernel_size, kernel_size))

    # Normalisation du noyau
    kernel /= np.sum(kernel) #On divise tout les elements par la somme des elements , pemret de maintenir la forme de la sitribution gaussienne ( lumiance, niveau de gris, ...)

    # Application du flou gaussien en utilisant une convolution
    blurred = convolve(image, kernel)
    return blurred

def sobel_filters(image):
    # Noyaux de Sobel pour la détection des gradients horizontaux et verticaux
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Application des filtres de Sobel pour calculer les gradients horizontaux et verticaux
    gradient_x = convolve(image, kernel_x) #Difference des intensités horizontales
    gradient_y = convolve(image, kernel_y) #Difference des intensités verticales

    return gradient_x, gradient_y

def gradient_magnitude(gradient_x, gradient_y):
    # Calcul du module du gradient
    magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    return magnitude

def gradient_direction(gradient_x, gradient_y):
    # Calcul de la direction du gradient
    direction = np.arctan2(gradient_y, gradient_x)

    return direction

def non_maximum_suppression(magnitude, direction):
    # Ajustement de la direction du gradient à des valeurs entre 0 et 180 degrés
    direction = np.rad2deg(direction) % 180

    # Dimensions de l'image
    height, width = magnitude.shape

    # Création d'une image vide pour stocker les contours supprimés
    suppressed = np.zeros_like(magnitude)

    for i in range(1, height-1):
        for j in range(1, width-1):
            angle = direction[i, j]

            # Comparaison des magnitudes voisines selon l'orientation du gradient
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                if (magnitude[i, j] >= magnitude[i, j-1]) and (magnitude[i, j] >= magnitude[i, j+1]):
                    suppressed[i, j] = magnitude[i, j]
            elif (22.5 <= angle < 67.5):
                if (magnitude[i, j] >= magnitude[i-1, j+1]) and (magnitude[i, j] >= magnitude[i+1, j-1]):
                    suppressed[i, j] = magnitude[i, j]
            elif (67.5 <= angle < 112.5):
                if (magnitude[i, j] >= magnitude[i-1, j]) and (magnitude[i, j] >= magnitude[i+1, j]):
                    suppressed[i, j] = magnitude[i, j]
            elif (112.5 <= angle < 157.5):
                if (magnitude[i, j] >= magnitude[i-1, j-1]) and (magnitude[i, j] >= magnitude[i+1, j+1]):
                    suppressed[i, j] = magnitude[i, j]

    return suppressed

def double_threshold(image, low_threshold, high_threshold):
    # Dimensions de l'image
    height, width = image.shape

    # Calcul des seuils minimaux et maximaux
    high = high_threshold
    low = low_threshold

    # Création d'une image vide pour stocker les contours après seuillage
    thresholded = np.zeros_like(image)

    # Parcours de l'image pour appliquer le seuillage
    for i in range(height):
        for j in range(width):
            pixel = image[i, j]

            # Application du seuil haut
            if pixel >= high:
                thresholded[i, j] = 255
            # Application du seuil bas
            elif pixel >= low:
                thresholded[i, j] = 50  # 50 représente les pixels à potentiel de contour
            # Pixels en dessous du seuil bas
            else:
                thresholded[i, j] = 0

    return thresholded

def edge_tracking(image):
    # Dimensions de l'image
    height, width = image.shape

    # Parcours de l'image pour effectuer le suivi des contours
    for i in range(1, height-1):
        for j in range(1, width-1):
            # Si un pixel à potentiel de contour est détecté
            if image[i, j] == 50:
                # Recherche des pixels voisins qui ont également un potentiel de contour
                neighbors = image[i-1:i+2, j-1:j+2] #Cerclz 3*3 autour
                if 255 in neighbors:
                    image[i, j] = 255
                else:
                    image[i, j] = 0

    return image

# Chargement de l'image
image = cv2.imread('voiture7.jpg')

# Conversion en niveaux de gris
gray = grayscale(image)

# Application du flou gaussien
blurred = gaussian_blur(gray, kernel_size=5, sigma=1.4)

# Calcul des gradients avec les filtres de Sobel
gradient_x, gradient_y = sobel_filters(blurred)

# Calcul du module du gradient
magnitude = gradient_magnitude(gradient_x, gradient_y)

# Calcul de la direction du gradient
direction = gradient_direction(gradient_x, gradient_y)

# Suppression des pixels non-maximaux
suppressed = non_maximum_suppression(magnitude, direction)

# Seuillage double
thresholded = double_threshold(suppressed, low_threshold=51, high_threshold=102)

# Suivi des contours
edges = edge_tracking(thresholded)

# Affichage de l'image avec les contours détectés
edges = np.uint8(edges)
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
