import argparse

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
                       characters):
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

            # Generate luminosity values
            luminosity_values = generate_luminosity_values(pixel_data,
                                                           brightness_method)

            # Map to ASCII characters
            ascii_art = map_brightness_to_ascii(luminosity_values, characters)

            # Create final ASCII output
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
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Suppress informational output.")
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
                                          brightness_method, characters)

        if ascii_output:
            if args.output:
                with open(args.output, "w") as f:
                    f.write(ascii_output)
                logger.info(f"Saved ASCII art to {args.output}")
            else:
                print(ascii_output)
