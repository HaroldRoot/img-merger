import argparse
import logging
import re

from PIL import Image
from colorama import init, Fore

init(autoreset=True)


class CustomFormatter(logging.Formatter):
    def format(self, record):
        prefix_color = {
            logging.INFO: Fore.GREEN,
            logging.ERROR: Fore.RED
        }.get(record.levelno, Fore.RESET)
        formatted_message = super().format(record)
        return f"{prefix_color}{formatted_message}{Fore.RESET}"


logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = CustomFormatter('[%(levelname)s] - %(message)s')
handler.setFormatter(formatter)

if not logger.hasHandlers():
    logger.addHandler(handler)


def concatenate_images_vertically(image_paths, output_path):
    images = []
    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            images.append(img)
        except FileNotFoundError:
            logging.error(f"Warning: File not found - {image_path}")
        except IOError:
            logging.error(f"Warning: Could not open image file - {image_path}")

    if not images:
        logging.error("No valid images to concatenate.")
        return

    widths, heights = zip(*(image.size for image in images))
    total_height = sum(heights)
    max_width = max(widths)

    new_image = Image.new('RGB', (max_width, total_height))

    y_offset = 0
    for image in images:
        new_image.paste(image, (0, y_offset))
        y_offset += image.height

    try:
        new_image.save(output_path)
        logging.info(f"Concatenated image saved to {output_path}")
    except IOError:
        logging.error("Could not save the concatenated image.")


def expand_patterns(patterns):
    expanded_list = []
    for pattern in patterns:

        match = re.search(r'\[(\d+)-(\d+)]', pattern)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            num_length = len(match.group(1))
            for i in range(start, end + 1):
                expanded_filename = re.sub(r'\[\d+-\d+]', f"{i:0{num_length}}",
                                           pattern)
                expanded_list.append(expanded_filename)
        else:
            match_single = re.search(r'\[(\d)-(\d)]', pattern)
            if match_single:
                start, end = int(match_single.group(1)), int(
                    match_single.group(2))
                for i in range(start, end + 1):
                    expanded_filename = re.sub(r'\[\d-\d]', f"{i}", pattern)
                    expanded_list.append(expanded_filename)
            else:
                expanded_list.append(pattern)
    return expanded_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate images vertically.")
    parser.add_argument("input_paths", nargs="+",
                        help="Input image paths or patterns")
    parser.add_argument("-o", "--output", required=True,
                        help="Output image path")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    output_path = args.output
    input_patterns = args.input_paths
    input_paths = expand_patterns(input_patterns)

    if not input_paths:
        logging.error("No valid image files found based on the input patterns.")
    else:
        concatenate_images_vertically(input_paths, output_path)
