import cv2
import numpy as np
import sys

# Chargement de l'image
edges = cv2.imread('edges6.png', 0)

#Modification de la profondeur de recherche:
new_limit = 5000  # Nouvelle limite de profondeur maximale de récursion
sys.setrecursionlimit(new_limit)

#Utilitaire
def distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2)
#Obtention des contours
height, width = edges.shape[:2]
contours = []
visited = []

def complet_visited(L): #Incrementation de la list permettant de ne pas refaire plusieurs contours
    for i in L:
        if i not in visited:
            visited.append(i)

def next_w_pixel(i, j, image, temp_visited, not_allowed):
    L = []
    mouv_adjacent = [(0, -1), (-1, 0), (0, 1), (1, 0)]  # Déplacements pour les pixels adjacents (haut, gauche, bas, droite)
    mouv_diagonal = [(-1, -1), (-1, 1), (1, -1), (1, 1)]  # Déplacements pour les pixels en diagonale
    adjacent_found = False

    no_move = temp_visited + not_allowed #Ces points ne comptent pas

    for mov in mouv_adjacent:
        a, b = mov
        if 0 <= i + a < height and 0 <= j + b < width:  # Vérification pixel dans l'image
            if image[i + a][j + b] == 255:
                if (i + a, j + b) not in no_move:
                    L.append((i + a, j + b))
                    adjacent_found = True

    if not adjacent_found:
        for mov in mouv_diagonal:
            a, b = mov
            if 0 <= i + a < height and 0 <= j + b < width:  # Vérification pixel dans l'image
                if image[i + a][j + b] == 255:
                    if (i + a, j + b) not in no_move and (i + a, j + b) not in visited:
                        L.append((i + a, j + b))

    return L

def extractContours(i, j, image, temp_visited, not_allowed):
    temp_visited.append((i, j))
    next = next_w_pixel(i, j, image, temp_visited, not_allowed)
    if len(next) == 0:
        contours.append(temp_visited)
        complet_visited(temp_visited)

    elif len(next) == 1:
        extractContours(next[0][0], next[0][1], image, temp_visited, not_allowed)

    else:
        for i in range(len(next)):
            t_visited = temp_visited.copy() #Copie du parcours commun
            t_not_allowed = not_allowed.copy()
            r_next = next.copy()
            r_next.remove(next[i])
            extractContours(next[i][0], next[i][1], image,t_visited, t_not_allowed + r_next)

def count_neighbour_pixels(i, j, image):
    count = 0
    height, width = image.shape

    # Coordonnées des 8 pixels voisins
    neighbours = [(i-1, j-1), (i-1, j), (i-1, j+1),
                  (i, j-1),             (i, j+1),
                  (i+1, j-1), (i+1, j), (i+1, j+1)]

    for neighbour in neighbours:
        x, y = neighbour
        if 0 <= x < height and 0 <= y < width:  # Vérification si les coordonnées sont valides
            if image[x][y] == 255:  # Vérification si le pixel est blanc
                count += 1

    return count

def is_extremity(i, j, image): #Extremité pour commencer la detection la plus proche
    count = count_neighbour_pixels(i,j,image)
    if count < 2:
        return True
    else :
        return False

def detectContours(image):
    for i in range(height):
        for j in range(width):
            if image[i][j] == 255 and (i, j) not in visited:
                if is_extremity(i, j, image):
                    extractContours(i, j, image, [], [])


detectContours(edges)

def sort_and_get_top_contours(contours):
    # Trier les contours par leur aire (taille)
    sorted_contours = sorted(contours, key=len, reverse=True)

    # Récupérer les 50 plus grands contours
    top_contours = sorted_contours[:20]

    return top_contours

print(len(contours))
contours = sort_and_get_top_contours(contours)
print("OK")
#Bounding box:
def bounding_box(contour):
    # min de y (ligne)
    min_row = min(contour, key=lambda x: x[0])[0] #key=lambda x: x[0] signifie que l'on trie selon la premiére coordonée (ici y)
    # max de y
    max_row = max(contour, key=lambda x: x[0])[0]

    # min de x
    min_col = min(contour, key=lambda x: x[1])[1]

    # max de x
    max_col = max(contour, key=lambda x: x[1])[1]

    # Calcul des dimensions du rectangle englobant
    width = max_col - min_col
    height = max_row - min_row

    return min_row, min_col, width, height # (min_row,max_row) les coords du point en haut a gauche

#Verification du fait qu'un contour soit fermé
def tracer_ligne(image, x1, y1, x2, y2):
    # Créer une copie de l'image pour ne pas modifier l'originale
    image_ligne = np.copy(image)

    # Tracer la ligne blanche
    cv2.line(image_ligne, (x1, y1), (x2, y2), (255), 1)

    # Calculer les coordonnées des points intermédiaires
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    steps = max(dx, dy)

    points_ligne = []

    if steps > 0:
        x_step = (x2 - x1) / steps
        y_step = (y2 - y1) / steps

        x = x1
        y = y1

        # Ajouter les points intermédiaires à la liste
        for _ in range(steps - 1):
            x += x_step
            y += y_step
            points_ligne.append((int(round(x)), int(round(y))))

    return points_ligne


def closed_contour(image,contours):
    c_contours = []
    for contour in contours:
        depart = contour[0]
        fin = contour[-1]
        dist = distance(depart[0],depart[1],fin[0],fin[1])

        if dist < (1/4) * len(contour):
            points_ligne = tracer_ligne(image,depart[0],depart[1],fin[0],fin[1])
            c_contours.append(contour + points_ligne)

    return c_contours


#Calcul de l'air d'un contour supposé fermé
def calculate_contour_area(contour):
    area = 0
    n = len(contour)

    for i in range(n):
        x1, y1 = contour[i]
        x2, y2 = contour[(i + 1) % n]  # Point suivant (bouclage au premier point)

        area += (x1 * y2 - x2 * y1)

    area = abs(area) / 2 #Ainsi l'air réel est positive
    return int(area)

#Test pour savoir si le contour est assimilable a sa box englobante (plus un contour est a la fin de la liste plus il est proche de sa box
def is_rectangle(image, c_contours):
    contours_proches = []

    for contour in c_contours:
        box = bounding_box(contour)
        box_area = box[2] * box[3]
        contour_area = calculate_contour_area(contour)

        proximity = abs(box_area - contour_area)
        contours_proches.append((contour, proximity))

    contours_proches = sorted(contours_proches, key=lambda x: x[1], reverse=False) #On trie selon la proximity d'ou x[1]
    sorted_contours = [contour for contour, _ in contours_proches]

    return sorted_contours

#Verification du ration
def is_plate(image,c_contours):
    plaque = None
    for contour in is_rectangle(image,c_contours):
        x, y, w, h = bounding_box(contour)
        if w > h :
            ratio = w/h   #ratio d'une plaque 52/11
            if abs(ratio-52/11) < 0.2*(52/11) : #Seuil de tolérance de 20 pourcents
                plaque = contour

    return plaque

#Extraction
def crop_image(image, x, y, w, h):
    # Récupérer la partie de l'image correspondant au rectangle
    cropped_image = image[y+5:y+h-5, x+5:x+w-5].copy() #On enleve 10 pixels pour ne plus avoir le contour de la plaque

    # Redimensionner l'image pour correspondre exactement à la taille du rectangle

    return cropped_image

#Utilisation
c_contours = closed_contour(edges,contours)

# image_with_points = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
# for contour in c_contours:
#     for pixel in contour:
#         cv2.circle(image_with_points, (pixel[1], pixel[0]), 0, (0, 255, 0), -1)  # Dessin des contours en vert
#
#     y, x, w, h = bounding_box(contour)
#     cv2.rectangle(image_with_points, (x, y), (x + w, y + h), (0, 0, 255), 3)
#
# cv2.imshow('Image avec contours', image_with_points)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

valid_contour = is_plate(edges,c_contours)

# Dessin des contours en vert avec une épaisseur de 1 pixel
image_with_contours = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
y, x, w, h = bounding_box(valid_contour)
cv2.rectangle(image_with_contours, (x, y), (x + w, y + h), (0, 0, 255), 3)

cv2.imshow('Image avec contours', image_with_contours)
cv2.waitKey(0)
cv2.destroyAllWindows()


y, x, w, h = bounding_box(valid_contour)
voiture = cv2.imread('voiture7.jpg', 0)
roi_image = crop_image(voiture, x, y, w, h)

# Affichage des images
cv2.imshow('Image avec contours', roi_image)
cv2.waitKey(0)
cv2.destroyAllWindows()



