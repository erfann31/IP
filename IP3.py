import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('lena.jpg', 0)

f = np.fft.fft2(img)
f = np.fft.fftshift(f)
absf = np.abs(f)
logabs = 200 * np.log(absf)
plt.subplot(221)
plt.imshow(img, cmap='gray')
plt.subplot(222)
plt.imshow(logabs, cmap='gray')

# #midpass
bw1 = 230
bw2 = 20
rows, cols = img.shape
crow, ccol = rows // 2, cols // 2
ze = np.zeros((rows, cols))
ze[crow - bw1:crow + bw1, ccol - bw1:ccol + bw1] = 1
ze[crow - bw2:crow + bw2, ccol - bw2:ccol + bw2] = 0
f1 = f * ze

absf = np.abs(f1)
logabs = 200 * np.log(absf + 1e-8)
plt.subplot(224)
plt.imshow(logabs, cmap='gray')

imrecover = np.fft.ifftshift(f1)
imrecover = np.fft.ifft2(imrecover)
image = np.abs(imrecover)

plt.subplot(223)
plt.imshow(image, cmap='gray')
image = image.astype(np.uint8)
cv2.imshow('res',image)
plt.show()
cv2.waitKey(0)