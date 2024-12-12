import argparse
import math
import random
import time

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from colorama import Fore
from colorama import init

from custom_logger import logger

init(autoreset=True)

characters = ("`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8"
              "%B@$")


# Brightness calculation methods
def calculate_brightness_average(pixel):
    return sum(pixel[:3]) // 3


def calculate_brightness_min_max(pixel):
    return (max(pixel) + min(pixel)) // 2


def calculate_brightness_luminosity(pixel):
    return int(0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2])


def invert_brightness(brightness):
    return 255 - brightness


# Convert pixel data to brightness values, with optional inversion
def generate_luminosity_values(pixel_data, brightness_method, invert=False):
    luminosity_values = [[brightness_method(pixel) for pixel in row] for row in
                         pixel_data]

    if invert:
        luminosity_values = [[invert_brightness(lum) for lum in row] for row in
                             luminosity_values]

    return luminosity_values


COLOR_MAP = {
    # Fore.BLACK: (0, 0, 0),
    Fore.RED: (255, 0, 0),
    Fore.GREEN: (0, 255, 0),
    Fore.YELLOW: (255, 255, 0),
    Fore.BLUE: (0, 0, 255),
    Fore.MAGENTA: (255, 0, 255),
    Fore.CYAN: (0, 255, 255),
    Fore.WHITE: (255, 255, 255)
}


def euclidean_distance(color1, color2):
    return math.sqrt(
        sum((int(c1) - int(c2)) ** 2 for c1, c2 in zip(color1, color2)))


def get_dominant_color(pixel):
    pixel_color = pixel[:3]
    distances = {color: euclidean_distance(pixel_color, ref_color) for
                 color, ref_color in COLOR_MAP.items()}
    return min(distances, key=distances.get)


def img2ascii_kmeans(frame, K=5, colorful=False):
    if len(frame.shape) == 2:
        frame = np.stack([frame] * 3, axis=-1)

    height, width, *_ = frame.shape
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_array = np.float32(frame_gray.reshape(-1))

    bestLabels = np.zeros((frame_array.shape[0], 1), dtype=np.int32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    _, labels, centroids = cv2.kmeans(frame_array, K, bestLabels, criteria, 10,
                                      flags)
    centroids = np.uint8(centroids)

    centroids_sorted = sorted(centroids.flatten())
    centroids_index = np.array(
        [centroids_sorted.index(value) for value in centroids.flatten()])

    labels = labels.flatten()
    labels = centroids_index[labels]
    labels_picked = [labels[rows * width:(rows + 1) * width] for rows in
                     range(height)]

    ascii_art = []
    for row_idx, rows in enumerate(labels_picked):
        row_art = ""
        for col_idx, col in enumerate(rows):
            char = "#" if col == K - 1 else "-" if col >= K // 2 else str(
                random.randint(2, 9))
            if colorful:
                pixel = frame[row_idx, col_idx]
                dominant_color = get_dominant_color(pixel)
                row_art += dominant_color + char * 2
            else:
                row_art += char * 2
        ascii_art.append(row_art)

    return "\n".join(ascii_art)


def map_brightness_to_ascii(luminosity_values, pixel_data, colorful=False):
    ascii_art = []
    scale = len(characters) - 1
    if colorful:
        for row_idx, row in enumerate(luminosity_values):
            row_art = ""
            for col_idx, lum in enumerate(row):
                pixel = pixel_data[row_idx][col_idx]
                dominant_color = get_dominant_color(pixel)
                c = characters[round((lum / 255) * scale)]
                row_art += dominant_color + c * 2
            ascii_art.append(row_art)
    else:
        for row in luminosity_values:
            ascii_art.append("".join(
                characters[round((lum / 255) * scale)] * 2 for lum in row))
    return "\n".join(ascii_art)


def img2ascii_brightness(img, width, height, brightness_method, invert=False,
                         colorful=False):
    pixel_data = [[img.getpixel((x, y)) for x in range(width)] for y in
                  range(height)]
    luminosity_values = generate_luminosity_values(pixel_data,
                                                   brightness_method, invert)
    return map_brightness_to_ascii(luminosity_values, pixel_data,
                                   colorful)


def pixel_data_to_rgb_ascii(pixel_data):
    scale = len(characters) - 1

    ascii_art = "\n".join(
        "".join(
            (
                f"{Fore.RED}{characters[round((pixel[0] / 255) * scale)]}"
                f"{Fore.GREEN}{characters[round((pixel[1] / 255) * scale)]}"
                f"{Fore.BLUE}{characters[round((pixel[2] / 255) * scale)]}"
                if isinstance(pixel, (tuple, list)) and len(pixel) >= 3 and all(
                    isinstance(c, (int, float)) for c in pixel[:3])
                else "   "  # 用空格占位以保证行列对齐
            )
            for pixel in row
        )
        for row in pixel_data
    )
    return ascii_art


def img2ascii_rgb(img, width, height):
    new_width = int(width * 2 / 3)
    img = img.resize((new_width, height), Image.Resampling.LANCZOS)

    pixel_data = [
        [img.getpixel((x, y)) for x in range(new_width)]
        for y in range(height)
    ]
    return pixel_data_to_rgb_ascii(pixel_data)


def generate_ascii_art(input_path, max_width, max_height, brightness_method,
                       kmeans=False, invert=False, colorful=False, K=5,
                       rgb=False, color=None):
    try:
        if input_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            return process_video_to_ascii(input_path, max_width // 3,
                                          max_height // 3, colorful)

        with Image.open(input_path) as img:
            img.thumbnail((max_width, max_height))
            width, height = img.size

            if rgb:
                return img2ascii_rgb(img, width, height)
            elif kmeans:
                frame = np.array(img)
                return img2ascii_kmeans(frame, K, colorful)
            else:
                ascii_art = img2ascii_brightness(img, width, height,
                                                 brightness_method, invert,
                                                 colorful)
                if color and color in COLOR_MAP:
                    ascii_art = "\n".join(
                        color + line for line in ascii_art.splitlines())
                return ascii_art

    except (FileNotFoundError, UnidentifiedImageError):
        logger.error(f"Error processing input: {input_path}")
        return None


def process_video_to_ascii(video_path, width, height, colorful=False, fps=10):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Resize frame for better performance
            frame = cv2.resize(frame, (width, height),
                               interpolation=cv2.INTER_LINEAR)

            # Convert to ASCII
            frame_ascii = img2ascii_kmeans(frame, K=5, colorful=colorful)

            # Clear screen and print ASCII art
            print("\033[H\033[J", end="")
            print(frame_ascii)

            # Control the frame rate
            time.sleep(1 / fps)

        cap.release()

    except Exception as e:
        logger.error(f"Error processing video: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Convert media to ASCII art.")
    parser.add_argument("input_paths", nargs="+",
                        help="Paths to input media files (images or videos).")
    parser.add_argument("-W", "--width", type=int, default=940,
                        help="Max width of output (default: 940).")
    parser.add_argument("-H", "--height", type=int, default=470,
                        help="Max height of output (default: 470).")
    parser.add_argument("-b", "--brightness",
                        choices=["average", "min_max", "luminosity"],
                        default="luminosity", help="Brightness mapping method "
                                                   "(default: luminosity).")
    parser.add_argument("-k", "--kmeans", action="store_true",
                        help="Use K-means clustering for ASCII art.")
    parser.add_argument("-c", "--colorful", action="store_true",
                        help="Enable colorful ASCII art.")
    parser.add_argument("-i", "--invert", action="store_true",
                        help="Invert brightness values.")
    parser.add_argument("--rgb", action="store_true",
                        help="Use RGB mode for ASCII art (colorful per pixel).")
    parser.add_argument("--color",
                        choices=["red", "green", "yellow", "blue", "magenta",
                                 "cyan", "white"],
                        help="Specify ASCII art color.")
    parser.add_argument("--fps", type=int, default=10,
                        help="Frames per second for ASCII animation "
                             "(default: 10).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    brightness_methods = {"average": calculate_brightness_average,
                          "min_max": calculate_brightness_min_max,
                          "luminosity": calculate_brightness_luminosity}
    brightness_method = brightness_methods[args.brightness]

    for input_path in args.input_paths:
        ascii_output = generate_ascii_art(input_path, args.width,
                                          args.height, brightness_method,
                                          args.kmeans, args.invert,
                                          args.colorful, rgb=args.rgb,
                                          color=getattr(Fore,
                                                        args.color.upper(),
                                                        None) if args.color
                                          else None)
        if ascii_output:
            print(ascii_output)
