import cv2
import numpy as np
import os

# 1️⃣ Charger l'image
image = cv2.imread("code0.tif", cv2.IMREAD_GRAYSCALE)

# 2️⃣ Appliquer un flou pour réduire le bruit
blurred = cv2.GaussianBlur(image, (5, 5), 0)

# 3️⃣ Binarisation avec Otsu
_, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# 4️⃣ Détection des contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 5️⃣ Définir le chemin vers ton dossier Python (modifie ce chemin si nécessaire)
output_folder =  "C:/Users/aboub/AppData/Local/Programs/Python/Python313/TP3 ISOLATION_CHIFFRES"
  # Mets le chemin exact de ton dossier

# 6️⃣ Vérifier si le dossier existe, sinon le créer
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 7️⃣ Sauvegarde des chiffres isolés dans ton dossier Python
for i, contour in enumerate(contours):
    x, y, w, h = cv2.boundingRect(contour)
    digit = thresh[y:y+h, x:x+w]
    cv2.imwrite(os.path.join(output_folder, f"chiffre_{i}.png"), digit)

print(f"Les chiffres isolés ont été enregistrés dans le dossier : {output_folder}")

# 8️⃣ Affichage des contours détectés
cv2.imshow("Contours détectés", thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
