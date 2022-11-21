import cv2
import os 
img = cv2.imread("spec.png")
dim = (224,224)
print(img.shape)
resized = cv2.resize(img,dim,interpolation = cv2.INTER_NEAREST)
print(resized.shape)
cv2.imwrite("spec-resize.png", resized)