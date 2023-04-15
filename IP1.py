import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
img = cv2.imread('8.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# Define the Sobel matrix for x and y directions
MX = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
MY = MX.transpose()

# Apply Sobel filter to the grayscale image in both x and y directions
sobelX = cv2.filter2D(gray, -1, MX)
sobelY = cv2.filter2D(gray, -1, MY)


# Convert Sobel filtered images to float64 data type for further processing
sobelX = np.float64(sobelX)
sobelY = np.float64(sobelY)

# Calculate the magnitude of the gradient using the Sobel filtered images
mag = np.sqrt(sobelX ** 2 + sobelY ** 2)

# Normalize the magnitude to a range between 0 and 255
mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# Display the Sobel filtered images after conversion and the normalized magnitude
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].imshow(gray, cmap='gray')
axs[0, 0].set_title("Original")

axs[0, 1].imshow(sobelX, cmap='gray')
axs[0, 1].set_title("Sobel X")

axs[1, 0].imshow(sobelY, cmap='gray')
axs[1, 0].set_title("Sobel Y")

axs[1, 1].imshow(mag, cmap='gray')
axs[1, 1].set_title("Sobel Magnitude")

plt.show()

# Wait indefinitely until a key is pressed and then close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
