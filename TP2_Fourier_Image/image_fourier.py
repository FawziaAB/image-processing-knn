import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, transform 

def plot_spectrum(f_transform, title):
    """Affiche le spectre de Fourier en échelle logarithmique."""
    
    magnitude_spectrum = 20 * np.log(np.abs(f_transform) + 1e-9)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title(title)
    plt.colorbar(label='Amplitude (dB)')
    plt.show()

# 1. Transformée de Fourier 2D
print("1. Transformée de Fourier 2D")

try:
    img_mit = io.imread('mit.tiff')
    img_cameraman = io.imread('cameraman.tif')
except FileNotFoundError:
    print("Erreur : Assurez-vous que 'mit.tiff' et 'cameraman.tif' sont dans le même dossier.")
    # Si les fichiers sont introuvables, vous pouvez tenter de les télécharger ou d'utiliser des chemins absolus.
    
    
# Convertir en niveaux de gris si les images sont en couleur
if len(img_mit.shape) == 3:
    img_mit = color.rgb2gray(img_mit)
if len(img_cameraman.shape) == 3:
    img_cameraman = color.rgb2gray(img_cameraman)

# Redimensionner l'image 'mit' pour qu'elle ait la même taille que 'cameraman'
target_shape = img_cameraman.shape # Prenez la taille de cameraman comme référence
img_mit_resized = transform.resize(img_mit, target_shape, anti_aliasing=True)

# Calculer la FFT 2D
f_mit = np.fft.fft2(img_mit_resized) # Utilisez l'image redimensionnée ici
f_cameraman = np.fft.fft2(img_cameraman)

# Translatation des basses fréquences au centre
f_mit_shifted = np.fft.fftshift(f_mit)
f_cameraman_shifted = np.fft.fftshift(f_cameraman)

# Afficher les images originales (redimensionnée pour mit)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(img_mit_resized, cmap='gray')
plt.title('Image "mit" (redimensionnée)')
plt.subplot(1, 2, 2)
plt.imshow(img_cameraman, cmap='gray')
plt.title('Image "cameraman"')
plt.show()


# Afficher les résultats des spectres
plot_spectrum(f_mit_shifted, 'Spectre de Fourier de "mit.tiff" (redimensionné, fréquences basses au centre)')
plot_spectrum(f_cameraman_shifted, 'Spectre de Fourier de "cameraman.tif" (fréquences basses au centre)')

# 2. TFD 2D et reconstitution
print("\n2. TFD 2D et reconstitution")

# a) Modifier la valeur des images transformées en annulant la phase de leurs coefficients.
print("a) Annulation de la phase des coefficients")

# Obtenir le module et la phase
magnitude_mit = np.abs(f_mit)
phase_mit = np.angle(f_mit)

magnitude_cameraman = np.abs(f_cameraman)
phase_cameraman = np.angle(f_cameraman)

# Image transformée avec phase nulle (conservation du module uniquement)
# La phase nulle signifie que la partie imaginaire est 0, et la partie réelle est le module.
# Note: Pour une phase nulle, le nombre complexe est simplement son module (partie réelle)
f_mit_phase_zero = magnitude_mit
f_cameraman_phase_zero = magnitude_cameraman

# Reconstitution de l'image
reconstructed_mit_phase_zero = np.fft.ifft2(f_mit_phase_zero)
reconstructed_cameraman_phase_zero = np.fft.ifft2(f_cameraman_phase_zero)

# Afficher les images reconstituées (prendre la partie réelle car ifft2 peut donner de petites parties imaginaires)
plt.imshow(np.real(reconstructed_mit_phase_zero), cmap='gray')
plt.title('Reconstitution "mit" (phase nulle)')
plt.show()

plt.imshow(np.real(reconstructed_cameraman_phase_zero), cmap='gray')
plt.title('Reconstitution "cameraman" (phase nulle)')
plt.show()


# b) Même question, en donnant le même module à tous les coefficients :
print("b) Module constant")

# Choisir un module constant (par exemple, la moyenne des modules)
# Il est important de choisir une valeur non nulle.
mean_magnitude_mit = np.mean(magnitude_mit)
mean_magnitude_cameraman = np.mean(magnitude_cameraman)

# Créer des matrices avec le module constant, en gardant la taille de l'image
constant_magnitude_mit = np.full_like(f_mit, mean_magnitude_mit)
constant_magnitude_cameraman = np.full_like(f_cameraman, mean_magnitude_cameraman)

# Recréer le signal de Fourier avec module constant et phase originale
f_mit_module_constant = constant_magnitude_mit * np.exp(1j * phase_mit)
f_cameraman_module_constant = constant_magnitude_cameraman * np.exp(1j * phase_cameraman)

# Reconstitution de l'image
reconstructed_mit_module_constant = np.fft.ifft2(f_mit_module_constant)
reconstructed_cameraman_module_constant = np.fft.ifft2(f_cameraman_module_constant)

# Afficher les images reconstituées
plt.imshow(np.real(reconstructed_mit_module_constant), cmap='gray')
plt.title('Reconstitution "mit" (module constant)')
plt.show()

plt.imshow(np.real(reconstructed_cameraman_module_constant), cmap='gray')
plt.title('Reconstitution "cameraman" (module constant)')
plt.show()

# 3. Création d’une image à partir du module et de la phase
print("\n3. Création d’une image à partir du module et de la phase")

# Phase de "mit" (maintenant de la même taille que cameraman)
phase_of_mit = np.angle(f_mit)

# Module de "cameraman" (maintenant de la même taille que mit)
magnitude_of_cameraman = np.abs(f_cameraman)

# Créer la nouvelle transformée de Fourier (maintenant, les deux opérandes ont la même taille)
new_f_transform = magnitude_of_cameraman * np.exp(1j * phase_of_mit)

# Reconstituer l'image
reconstructed_hybrid_image = np.fft.ifft2(new_f_transform)

# Afficher l'image reconstituée
plt.imshow(np.real(reconstructed_hybrid_image), cmap='gray')
plt.title('Image reconstituée (Module de Cameraman, Phase de Mit)')
plt.show()

