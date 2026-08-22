import cv2
import os


def print_image_information(image):
    height, width, channels = image.shape
    print("height:", height)
    print("width:", width)
    print("channels:", channels)
    print("size:", image.size)
    print("data type:", image.dtype)


def save_camera_information(output_path="solutions/camera_outputs.txt"):
    cam = cv2.VideoCapture(0)

    fps = cam.get(cv2.CAP_PROP_FPS)
    width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam.release()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(f"fps: {fps}\n")
        f.write(f"height: {height}\n")
        f.write(f"width: {width}\n")


def main():
    image = cv2.imread("iris-1.jpg")
    print_image_information(image)
    save_camera_information()


if __name__ == "__main__":
    main()