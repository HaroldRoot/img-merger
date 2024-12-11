import argparse
import math
import random

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
def generate_luminosity_values(pixel_data, invert=False):
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
    color1 = [int(c) for c in color1]
    color2 = [int(c) for c in color2]
    return math.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2)))


def get_dominant_color(pixel):
    pixel_color = pixel[:3]
    distances = {color: euclidean_distance(pixel_color, ref_color) for
                 color, ref_color in COLOR_MAP.items()}
    dominant_color = min(distances, key=distances.get)
    return dominant_color


# K-means clustering method
def img2ascii_kmeans(frame, K=5, colorful=False):
    if len(frame.shape) == 2:
        frame = np.stack([frame] * 3, axis=-1)

    height, width, *_ = frame.shape
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_array = np.float32(frame_gray.reshape(-1))

    bestLabels = np.zeros((frame_array.shape[0], 1), dtype=np.int32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    compactness, labels, centroids = cv2.kmeans(frame_array, K, bestLabels,
                                                criteria, 10, flags)
    centroids = np.uint8(centroids)

    centroids_sorted = sorted(centroids.flatten())
    centroids_index = np.array(
        [centroids_sorted.index(value) for value in centroids.flatten()])

    bright = [abs((3 * i - 2 * K) / (3 * K)) for i in range(1, 1 + K)]
    bright_bound = bright.index(np.min(bright))
    shadow = [abs((3 * i - K) / (3 * K)) for i in range(1, 1 + K)]
    shadow_bound = shadow.index(np.min(shadow))

    labels = labels.flatten()
    labels = centroids_index[labels]

    labels_picked = [labels[rows * width:(rows + 1) * width] for rows in
                     range(height)]

    ascii_art = []
    for row_idx, rows in enumerate(labels_picked):
        row_art = ""
        for col_idx, col in enumerate(rows):
            if col <= shadow_bound:
                c = str(random.randint(2, 9))
            elif col <= bright_bound:
                c = "-"
            else:
                c = "#"

            if colorful:
                # Extract RGB color of the pixel
                pixel = frame[row_idx, col_idx]
                dominant_color = get_dominant_color(pixel)
                row_art += dominant_color + c + c  # Add color and character
            else:
                row_art += c + c

        ascii_art.append(row_art)

    return "\n".join(ascii_art)


def map_brightness_to_ascii(luminosity_values, pixel_data, colorful=False):
    ascii_art = []
    scale = len(characters) - 1
    if colorful:
        for row_idx, row in enumerate(luminosity_values):
            row_art = ""
            for col_idx, lum in enumerate(row):
                # Get the dominant color for the pixel
                pixel = pixel_data[row_idx][col_idx]
                dominant_color = get_dominant_color(pixel)

                # If there's a dominant color, use that for the printout
                if dominant_color:
                    c = characters[round((lum / 255) * scale)]
                    row_art += dominant_color + c + c
                else:
                    # For grayscale, use the calculated brightness for the ASCII
                    # character
                    c = characters[round((lum / 255) * scale)]
                    row_art += Fore.WHITE + c + c
            ascii_art.append(row_art)
        ascii_art = "\n".join(ascii_art)
    else:
        for row in luminosity_values:
            ascii_art.append(
                [characters[round((lum / 255) * scale)] for lum in row])
        ascii_art = "\n".join(["".join(char * 2 for char in row) for row in
                               ascii_art])

    return ascii_art


def img2ascii_brightness(img, width, height, invert=False, colorful=False):
    pixel_data = [[img.getpixel((x, y)) for x in range(width)] for y in
                  range(height)]
    luminosity_values = generate_luminosity_values(pixel_data, invert)
    ascii_art = map_brightness_to_ascii(luminosity_values, pixel_data, colorful)
    return ascii_art


# Generate ASCII art from an image
def generate_ascii_art(max_width, max_height, kmeans=False, invert=False,
                       colorful=False, K=5):
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            logger.info(f"Loaded image: {image_path} "
                        f"({original_width}x{original_height})")

            img.thumbnail((max_width, max_height))
            width, height = img.size
            logger.info(f"Resized image to: {width}x{height}")

            if kmeans:
                frame = np.array(img)
                return img2ascii_kmeans(frame, K, colorful)
            else:
                return img2ascii_brightness(img, width, height, invert,
                                            colorful)

    except FileNotFoundError:
        logger.error(f"Image file not found: {image_path}")
    except UnidentifiedImageError:
        logger.error(f"Invalid image file: {image_path}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Convert image to ASCII art.")
    parser.add_argument("input_paths", nargs="+",
                        help="Paths to input image files.")
    parser.add_argument("-W", "--width", type=int, default=470,
                        help="Max width of output (default: 470).")
    parser.add_argument("-H", "--height", type=int, default=235,
                        help="Max height of output (default: 235).")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to save ASCII art to a file.")
    parser.add_argument("-b", "--brightness",
                        choices=["average", "min_max", "luminosity"],
                        default="luminosity", help="Brightness mapping method "
                                                   "(default: luminosity).")
    parser.add_argument("-k", "--kmeans", action="store_true",
                        help="Use K-means clustering to generate ASCII art.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress informational output.")
    parser.add_argument("-c", "--color",
                        choices=["red", "green", "yellow", "blue", "magenta",
                                 "cyan", "white"], default="white",
                        help="Color for the ASCII output (default: white).")
    parser.add_argument("-f", "--colorful", action="store_true",
                        help="Enable colorful ASCII art.")
    parser.add_argument("-i", "--invert", action="store_true",
                        help="Invert brightness values (dark becomes light "
                             "and vice versa).")
    return parser.parse_args()


color_map = {"red": Fore.RED, "green": Fore.GREEN, "yellow": Fore.YELLOW,
             "blue": Fore.BLUE, "magenta": Fore.MAGENTA, "cyan": Fore.CYAN,
             "white": Fore.WHITE}

if __name__ == "__main__":
    args = parse_args()

    # Configure logger verbosity
    if args.quiet:
        logger.setLevel("ERROR")

    brightness_methods = {"average": calculate_brightness_average,
                          "min_max": calculate_brightness_min_max,
                          "luminosity": calculate_brightness_luminosity}

    brightness_method = brightness_methods[args.brightness]

    for image_path in args.input_paths:
        ascii_output = generate_ascii_art(args.width, args.height,
                                          kmeans=args.kmeans,
                                          invert=args.invert,
                                          colorful=args.colorful)

        if ascii_output:
            color = color_map[args.color]
            colored_output = color + ascii_output

            if args.output:
                with open(args.output, "w") as f:
                    f.write(ascii_output)
                logger.info(f"Saved ASCII art to {args.output}")
            else:
                print(colored_output)
