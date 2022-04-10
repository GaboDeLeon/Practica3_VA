import cv2
import numpy as np
import matplotlib.pyplot as plt

img1 = cv2.imread('img1.jpg',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img2.jpg',cv2.IMREAD_GRAYSCALE)
suma = cv2.addWeighted(img1, 0.5, img2, 0.5, 1)

def eq(img):
    width, height = img.shape
    x = np.linspace(0, 255, num = 256, dtype = np.uint8)
    y = np.zeros(256)
    equalized = np.zeros(img.shape, img.dtype)
    
    for w in range(width):
        for h in range(height):
            aux = img[w, h]
            y[aux] = y[aux] + 1
            
    k = 255/(width*height)
    _sum = 0
    
    for w in range(width):
        for h in range(height):
            for s in range(img[w, h]):
                _sum = _sum + y[s]
            equalized[w, h] = k*_sum
            _sum = 0
            
    return equalized

eq_1 = eq(img1)
eq_sum = eq(suma)
eq_2 = eq(img2)

#///////////////////////////imagen1//////
cv2.imshow('imagen 1',img1)

hist1 = cv2.calcHist(img1,[0],None, [256], [0,256])
plt.plot(hist1, color='gray')
plt.xlabel('imagen 1')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('equalizado 1',eq_1)

hist1_eq = cv2.calcHist(eq_1,[0],None, [256], [0,256])
plt.plot(hist1_eq, color='gray')
plt.xlabel('equalizado 1')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

#/////////////////////////////suma///////
cv2.imshow('suma',suma)

histSuma = cv2.calcHist(suma,[0],None, [256], [0,256])
plt.plot(histSuma, color='gray')
plt.xlabel('suma')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('equalizado suma',eq_sum)

histSuma_eq = cv2.calcHist(eq_sum,[0],None, [256], [0,256])
plt.plot(histSuma_eq, color='gray')
plt.xlabel('equalizado suma')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()


#////////////////////////////imagen2//////
cv2.imshow('imagen 2',img2)

hist2 = cv2.calcHist(img2,[0],None, [256], [0,256])
plt.plot(hist2, color='gray')
plt.xlabel('imagen 2')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('equalizado 2',eq_2)

hist2_eq = cv2.calcHist(eq_2,[0],None, [256], [0,256])
plt.plot(hist2, color='gray')
plt.xlabel('equalizado 2')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

