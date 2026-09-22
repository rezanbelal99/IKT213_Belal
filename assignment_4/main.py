import os

import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def harris(reference_image):
    img = reference_image.copy()
    gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imwrite(os.path.join(BASE_DIR, "harris.png"), img)


def align_images(image_to_align, reference_image, max_features, good_match_precent):
    gray_align = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray_ref = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_align, None)
    kp2, des2 = sift.detectAndCompute(gray_ref, None)

    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    matches = flann.knnMatch(des1, des2, k=2)

    good = [
        pair[0]
        for pair in matches
        if len(pair) == 2 and pair[0].distance < good_match_precent * pair[1].distance
    ]

    if len(good) <= max_features:
        print(f"Not enough matches found: {len(good)}/{max_features}")
        return

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    h, w = reference_image.shape[:2]
    aligned = cv2.warpPerspective(image_to_align, H, (w, h))

    matches_img = cv2.drawMatches(  # type: ignore
        image_to_align, kp1,
        reference_image, kp2,
        good, None,
        matchColor=(0, 255, 0),
        matchesMask=mask.ravel().tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    cv2.imwrite(os.path.join(BASE_DIR, "aligned.png"), aligned)
    cv2.imwrite(os.path.join(BASE_DIR, "matches.png"), matches_img)


if __name__ == "__main__":
    reference = cv2.imread(os.path.join(BASE_DIR, "reference_img.png"))
    to_align = cv2.imread(os.path.join(BASE_DIR, "align_this.jpg"))

    harris(reference)
    align_images(to_align, reference, 10, 0.7)