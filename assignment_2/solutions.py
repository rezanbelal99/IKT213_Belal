import cv2
import numpy as np


img = cv2.imread("iris.png")

if img is None:
    raise ValueError("Could not read image")

def padding(image, border_width):
    padded = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv2.BORDER_REFLECT)
    return padded


padded = padding(img, 100)

cv2.imwrite("padded.png", padded)


def crop(image, x_0, x_1, y_0, y_1):
    cropped = image[y_0:y_1, x_0:x_1]
    return cropped
height, width = img.shape[:2]

cropped = crop(img, 200, width - 130, 200, height - 130)
cv2.imwrite("cropped.png", cropped)


def resize(image, width, height):
    return cv2.resize(image, (width, height))

resized = resize(img, 200, 200)

cv2.imwrite("resized.png", resized)


def copy(image, emptyPictureArray):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            emptyPictureArray[y, x] = image[y, x]
    return emptyPictureArray

emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
copied = copy(img, emptyPictureArray)
cv2.imwrite("copy.png", copied)

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gscale = grayscale(img)
cv2.imwrite("grayscale.png", gscale)

def hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv_img = hsv(img)
cv2.imwrite("hsv.png", hsv_img)

def hue_shifted(image, emptyPictureArray, hue):
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            emptyPictureArray[y, x] = np.clip(image[y, x].astype(np.int16) + hue, 0, 255)
    return emptyPictureArray


hueArray = np.zeros((height, width, 3), dtype=np.uint8)

hue_shift = hue_shifted(img, hueArray, 50)
cv2.imwrite("hue_shifted.png", hue_shift)

def smoothing(image):
    return cv2.GaussianBlur(image, (15, 15), 0)
smoothed = smoothing(img)
cv2.imwrite("smoothed.png", smoothed)

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    else:
        raise ValueError("Could not rotate image")

rotationArray = rotation(img, 180)
cv2.imwrite("rotation.png", rotationArray)