import cv2
import numpy as np


def sobel_edge_detection(image):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    sobel = cv2.Sobel(src=blurred, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    sobel = cv2.convertScaleAbs(sobel)
    cv2.imwrite("sobel.png", sobel)
    return sobel


def canny_edge_detection(image, threshold_1, threshold_2):
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    edges = cv2.Canny(blurred, threshold_1, threshold_2)
    cv2.imwrite("canny.png", edges)
    return edges


def template_match(image, template):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    h, w = template_gray.shape

    result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where(result >= threshold)

    matched = image.copy()
    for pt in zip(*loc[::-1]):
        cv2.rectangle(matched, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    cv2.imwrite("template_match.png", matched)
    return matched


def resize(image, scale_factor: int, up_or_down: str):
    resized = image.copy()

    if up_or_down == "up":
        for _ in range(scale_factor):
            resized = cv2.pyrUp(resized)
        cv2.imwrite("resize_up.png", resized)
    elif up_or_down == "down":
        for _ in range(scale_factor):
            resized = cv2.pyrDown(resized)
        cv2.imwrite("resize_down.png", resized)
    else:
        raise ValueError("up_or_down must be 'up' or 'down'")

    return resized


if __name__ == "__main__":
    lambo = cv2.imread("lambo.png")
    shapes = cv2.imread("shapes.png")
    shapes_template = cv2.imread("shapes_template.jpg")

    if lambo is None or shapes is None or shapes_template is None:
        raise FileNotFoundError("One or more images failed to load")

    sobel_edge_detection(lambo)
    canny_edge_detection(lambo, 50, 50)
    template_match(shapes, shapes_template)
    resize(lambo, 2, "up")
    resize(lambo, 2, "down")