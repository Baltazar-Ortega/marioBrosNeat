import cv2
 
# Cargamos la imagen del disco duro
imagen = cv2.imread("logo.jpg")
 
cv2.imshow("prueba", imagen)
cv2.waitKey(0)