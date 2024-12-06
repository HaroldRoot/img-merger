import argparse
import random

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from custom_logger import logger


# Brightness calculation methods
def calculate_brightness_average(pixel):
    return sum(pixel) // 3


def calculate_brightness_lightness(pixel):
    return (max(pixel) + min(pixel)) // 2


def calculate_brightness_luminosity(pixel):
    return int(0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2])


# Convert pixel data to brightness values
def generate_luminosity_values(pixel_data, brightness_method):
    return [[brightness_method(pixel) for pixel in row] for row in pixel_data]


# K-means clustering method
def img2ascii_kmeans(frame, K=5, display=False):
    # Handle grayscale images
    if len(frame.shape) == 2:  # If the image is grayscale
        frame = np.stack([frame] * 3, axis=-1)  # Convert to 3-channel image

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

    labels_picked = [labels[rows * width:(rows + 1) * width:2] for rows in
                     range(0, height, 2)]

    # Prepare canvas
    canvas = np.zeros((3 * height, 3 * width, 3), np.uint8)
    canvas.fill(255)

    y = 8

    for rows in labels_picked:
        x = 0
        for cols in rows:
            if cols <= shadow_bound:
                random_digit = str(random.randint(2, 9))
                cv2.putText(canvas, random_digit, (x, y),
                            cv2.FONT_HERSHEY_PLAIN, 0.45, (0, 0, 0), 1)
            elif cols <= bright_bound:
                cv2.putText(canvas, "-", (x, y), cv2.FONT_HERSHEY_PLAIN, 0.4,
                            (0, 0, 0), 0, 1)
            x += 6
        y += 6

    if display:
        # Display the canvas instead of returning it
        cv2.imshow("Canvas", canvas)  # Display the image in a window
        cv2.waitKey(0)  # Wait until a key is pressed
        cv2.destroyAllWindows()  # Close the window after key press

    return None


# Convert brightness values to ASCII characters
def map_brightness_to_ascii(luminosity_values, characters):
    ascii_art = []
    scale = len(characters) - 1
    for row in luminosity_values:
        ascii_art.append(
            [characters[round((lum / 255) * scale)] for lum in row])
    return ascii_art


# Generate ASCII art from an image
def generate_ascii_art(image_path, max_width, max_height, brightness_method,
                       characters, kmeans=False, display=False):
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            logger.info(f"Loaded image: {image_path} "
                        f"({original_width}x{original_height})")

            # Resize the image
            img.thumbnail((max_width, max_height))
            width, height = img.size
            logger.info(f"Resized image to: {width}x{height}")

            # Extract pixel data
            pixel_data = [[img.getpixel((x, y)) for x in range(width)] for y in
                          range(height)]

            if kmeans:
                img_array = np.array(img)
                img2ascii_kmeans(img_array, K=5, display=display)
                return None
            else:
                luminosity_values = generate_luminosity_values(pixel_data,
                                                               brightness_method)
                ascii_art = map_brightness_to_ascii(luminosity_values,
                                                    characters)
                ascii_output = "\n".join(
                    ["".join(char * 2 for char in row) for row in ascii_art])

            logger.info("ASCII art generation complete.")
            return ascii_output

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
    parser.add_argument("-W", "--width", type=int, default=160,
                        help="Max width of output (default: 160).")
    parser.add_argument("-H", "--height", type=int, default=90,
                        help="Max height of output (default: 90).")
    parser.add_argument("-o", "--output", type=str,
                        help="Path to save ASCII art to a file.")
    parser.add_argument("-b", "--brightness",
                        choices=["average", "lightness", "luminosity"],
                        default="luminosity",
                        help="Brightness calculation method "
                             "(default: luminosity).")
    parser.add_argument("-k", "--kmeans", action="store_true",
                        help="Use K-means clustering to generate ASCII art.")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress informational output.")
    parser.add_argument("-d", "--display", action="store_true",
                        help="Display the image in a window (default: False).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Configure logger verbosity
    if args.quiet:
        logger.setLevel("ERROR")

    brightness_methods = {"average": calculate_brightness_average,
                          "lightness": calculate_brightness_lightness,
                          "luminosity": calculate_brightness_luminosity}

    characters = ("`^\",:;Il!i~+_-?][}{1)("
                  "|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$")
    brightness_method = brightness_methods[args.brightness]

    for image_path in args.input_paths:
        ascii_output = generate_ascii_art(image_path, args.width, args.height,
                                          brightness_method, characters,
                                          kmeans=args.kmeans,
                                          display=args.display)

        if ascii_output:
            if args.output:
                with open(args.output, "w") as f:
                    f.write(ascii_output)
                logger.info(f"Saved ASCII art to {args.output}")
            else:
                print(ascii_output)
