import cv2


def apply_clahe_filter(image_path):
    # Load the image
    img = cv2.imread(image_path)

    # Convert the image from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Split the HSV image into 3 channels
    h, s, v = cv2.split(hsv)

    # Apply CLAHE filter on the Value channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_clahe = clahe.apply(v)

    # Merge the updated V channel with the original H and S channels
    hsv_clahe = cv2.merge((h, s, v_clahe))

    # Convert the HSV image back to BGR
    bgr_clahe = cv2.cvtColor(hsv_clahe, cv2.COLOR_HSV2BGR)

    # Display the original and processed images
    cv2.imshow("Original Image", img)
    cv2.imshow("CLAHE Filtered Image", bgr_clahe)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Example usage of the function
apply_clahe_filter("lena.jpg")
