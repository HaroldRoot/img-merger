import argparse
import random

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError
from colorama import init, Fore

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


# Convert pixel data to brightness values
def generate_luminosity_values(pixel_data):
    return [[brightness_method(pixel) for pixel in row] for row in pixel_data]


# K-means clustering method
def img2ascii_kmeans(frame, K=5):
    if len(frame.shape) == 2:
        frame = np.stack([frame] * 3, axis=-1)

    height, width, *_ = frame.shape
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_array = np.float32(frame_gray.reshape(-1))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS

    compactness, labels, centroids = cv2.kmeans(frame_array, K, None, criteria,
                                                10, flags)
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

    # Generate ASCII characters for the image
    ascii_art = []
    for rows in labels_picked:
        row_art = ""
        for col in rows:
            if col <= shadow_bound:
                row_art += str(random.randint(2, 9))  # Shadow clusters
            elif col <= bright_bound:
                row_art += "-"  # Mid-clusters
            else:
                row_art += "#"  # Bright clusters
        ascii_art.append(row_art)

    # Return the final ASCII art as a string
    return "\n".join(["".join(char * 2 for char in row) for row in ascii_art])


# Convert brightness values to ASCII characters
def map_brightness_to_ascii(luminosity_values):
    ascii_art = []
    scale = len(characters) - 1
    for row in luminosity_values:
        ascii_art.append(
            [characters[round((lum / 255) * scale)] for lum in row])
    return ascii_art


def img2ascii_brightness(img, width, height):
    pixel_data = [[img.getpixel((x, y)) for x in range(width)] for y in
                  range(height)]
    luminosity_values = generate_luminosity_values(pixel_data)
    ascii_art = map_brightness_to_ascii(luminosity_values)
    return "\n".join(["".join(char * 2 for char in row) for row in ascii_art])


# Generate ASCII art from an image
def generate_ascii_art(max_width, max_height, kmeans=False):
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
                return img2ascii_kmeans(frame, K=5)
            else:
                return img2ascii_brightness(img, width, height)

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
                        default="luminosity",
                        help="Brightness mapping method "
                             "(default: luminosity).")
    parser.add_argument("-k", "--kmeans", action="store_true",
                        help="Use K-means clustering to generate ASCII art.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress informational output.")
    parser.add_argument("-c", "--color",
                        choices=["red", "green", "yellow", "blue", "magenta",
                                 "cyan", "white"], default="white",
                        help="Color for the ASCII output (default: white).")
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
                                          kmeans=args.kmeans)

        if ascii_output:
            color = color_map[args.color]
            colored_output = color + ascii_output

            if args.output:
                with open(args.output, "w") as f:
                    f.write(ascii_output)
                logger.info(f"Saved ASCII art to {args.output}")
            else:
                print(colored_output)
