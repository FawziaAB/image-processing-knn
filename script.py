import cv2
import numpy as np
import matplotlib.pyplot as plt

# Charger l'image en niveaux de gris
image = cv2.imread('spine.tif', cv2.IMREAD_GRAYSCALE)

if image is None:
    print(" L'image n'a pas été trouvée ! Vérifie son nom et son emplacement.")
else:
    # Calculer les histogrammes
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    hist_cumule = hist.cumsum()

    # Affichage des histogrammes
    plt.figure(figsize=(10,5))

    plt.subplot(1,2,1)
    plt.bar(range(256), hist, color='gray')
    plt.title("Histogramme d’amplitude")

    plt.subplot(1,2,2)
    plt.plot(hist_cumule, color='blue')
    plt.title("Histogramme cumulé")

    plt.show()

    

# Fonction de recadrage de l’image
def recadrage(image, a=None, b=None):
    if a is None:
        a = np.min(image)
    if b is None:
        b = np.max(image)

    # Recadrer l’image sur la plage [a, b]
    image_recadree = np.clip(image, a, b)

    # Normaliser entre 0 et 255
    image_recadree = ((image_recadree - a) / (b - a)) * 255
    image_recadree = image_recadree.astype(np.uint8)

    return image_recadree

# Charger l'image en niveaux de gris
image = cv2.imread('spine.tif', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("L'image n'a pas été trouvée ! Vérifie son nom et son emplacement.")
else:
    # Appliquer le recadrage avec `a=50` et `b=200`
    image_recadree = recadrage(image, a=50, b=200)

    # Affichage des images avant/après
    plt.figure(figsize=(10,5))
    
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.title("Image originale")

    plt.subplot(1,2,2)
    plt.imshow(image_recadree, cmap='gray')
    plt.title("Image recadrée (a=50, b=200)")

    plt.show()

    

# Fonction pour afficher en blanc les valeurs entre `a` et `b`
def affichage_blanc(image, a, b, f):
    # Création d'une copie de l'image
    image_transformee = image.copy()

    # Mettre en blanc les valeurs comprises entre `a` et `b`
    image_transformee[(image >= a) & (image <= b)] = 255

    if f == 0:
        # Éliminer le reste (mettre à noir)
        image_transformee[(image < a) | (image > b)] = 0

    return image_transformee

# Charger l'image en niveaux de gris
image = cv2.imread('spine.tif', cv2.IMREAD_GRAYSCALE)

if image is None:
    print(" L'image n'a pas été trouvée ! Vérifie son nom et son emplacement.")
else:
    # Appliquer la transformation avec `a=80, b=180` et `f=1`
    image_blanc = affichage_blanc(image, a=80, b=180, f=1)

    # Affichage des images avant/après
    plt.figure(figsize=(10,5))
    
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.title("Image originale")

    plt.subplot(1,2,2)
    plt.imshow(image_blanc, cmap='gray')
    plt.title("Valeurs entre a=80 et b=180 en blanc")

    plt.show()

    

# Fonction de dilatation et contraction autour de `A`
def dilatation_contraction(image, A, ad):
    # Vérification de la contrainte ad + ac = 1
    ac = 1 - ad

    # Création d'une copie de l'image
    image_modifiee = image.copy()

    # Appliquer la dilatation (augmentation des valeurs proches de A)
    image_modifiee[(image >= A)] = image_modifiee[(image >= A)] + int(ad * 70)
    
    # Appliquer la contraction (réduction des valeurs proches de A)
    image_modifiee[(image < A)] = image_modifiee[(image < A)] - int(ac * 70)

    # Clip pour garder les valeurs dans [0,255]
    image_modifiee = np.clip(image_modifiee, 0, 255)

    return image_modifiee.astype(np.uint8)

# Charger l’image en niveaux de gris
image = cv2.imread('spine.tif', cv2.IMREAD_GRAYSCALE)

if image is None:
    print(" L'image n'a pas été trouvée ! Vérifie son nom et son emplacement.")
else:
    # Appliquer la transformation avec `A=100` et `ad=0.9`
    image_dilatation = dilatation_contraction(image, A=100, ad=0.9)

    # Affichage des images avant/après
    plt.figure(figsize=(10,5))
    
    plt.subplot(1,2,1)
    plt.imshow(image, cmap='gray')
    plt.title("Image originale")

    plt.subplot(1,2,2)
    plt.imshow(image_dilatation, cmap='gray')
    plt.title("Image après dilatation/contraction (A=100, ad=0.9)")

    plt.show()


# Fonction d'égalisation d'histogramme
def egalisation_histogramme(image):
    # Appliquer l'égalisation d'histogramme avec OpenCV
    image_egalisee = cv2.equalizeHist(image)
    return image_egalisee

# Charger l'image en niveaux de gris
image = cv2.imread('spine.tif', cv2.IMREAD_GRAYSCALE)

if image is None:
    print(" L'image n'a pas été trouvée ! Vérifie son nom et son emplacement.")
else:
    # Appliquer l'égalisation d'histogramme
    image_egalisee = egalisation_histogramme(image)

    # Calculer les histogrammes avant et après égalisation
    hist_original, bins = np.histogram(image.flatten(), 256, [0, 256])
    hist_egalise, bins = np.histogram(image_egalisee.flatten(), 256, [0, 256])

    # Affichage des images et histogrammes
    plt.figure(figsize=(12,6))

    # Images
    plt.subplot(2,2,1)
    plt.imshow(image, cmap='gray')
    plt.title("Image originale")

    plt.subplot(2,2,2)
    plt.imshow(image_egalisee, cmap='gray')
    plt.title("Image égalisée")

    # Histogrammes
    plt.subplot(2,2,3)
    plt.plot(hist_original, color='gray')
    plt.title("Histogramme original")

    plt.subplot(2,2,4)
    plt.plot(hist_egalise, color='blue')
    plt.title("Histogramme égalisé")

    plt.show()