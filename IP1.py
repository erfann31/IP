import cv2
import numpy as np


def sobel_filter(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Sobel filter in x direction
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)

    # Apply Sobel filter in y direction
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate the norm of the two Sobel-filtered images
    sobel_norm = np.sqrt(np.square(sobel_x) + np.square(sobel_y))

    # Normalize the image
    sobel_norm = cv2.normalize(sobel_norm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Return the Sobel filtered images
    return sobel_x, sobel_y, sobel_norm


# Load the image
img = cv2.imread('lena.jpg')

# Apply the Sobel filter
sobel_x, sobel_y, sobel_norm = sobel_filter(img)

# Display the results
cv2.imshow('Sobel X', sobel_x)
cv2.imshow('Sobel Y', sobel_y)
cv2.imshow('Sobel Norm', sobel_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()