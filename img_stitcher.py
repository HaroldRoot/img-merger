import argparse
import logging
import re
from pathlib import Path

from PIL import Image
from colorama import init, Fore
from rich.prompt import Prompt

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


def concatenate_images_vertically(image_paths, output_path, width):
    images = []
    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            images.append(img)
        except FileNotFoundError:
            logging.error(f"File not found - {image_path}")
        except IOError:
            logging.error(f"Could not open image file - {image_path}")

    if not images:
        logging.error("No valid images to concatenate.")
        return

    widths, heights = zip(*(image.size for image in images))
    # If no width is provided, use the width of the first image
    if not width:
        width = widths[0]
    total_height = sum(
        int(height * (width / img_width)) for img_width, height in zip(
            widths, heights))

    new_image = Image.new('RGB', (width, total_height))

    y_offset = 0
    for image in images:
        # Adjust each image's width while maintaining aspect ratio
        aspect_ratio = image.height / image.width
        resized_height = int(width * aspect_ratio)
        resized_image = image.resize((width, resized_height),
                                     Image.Resampling.LANCZOS)

        # Paste the resized image onto the new_image canvas
        new_image.paste(resized_image, (0, y_offset))
        y_offset += resized_height

    try:
        new_image.save(output_path)
        abs_output_path = Path(output_path).resolve()
        logging.info(f"Concatenated image saved to {abs_output_path}")
    except IOError:
        logging.error("Could not save the concatenated image.")


def handle_output_pattern(pattern, ext):
    if "\\" in pattern:
        output_dir = Path(pattern).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    if not pattern.endswith(f".{ext}"):
        pattern += f".{ext}"
    return pattern


def batch_concatenate_images(image_files, output_pattern, n, ext, width):
    output_pattern = handle_output_pattern(output_pattern, ext)
    while Path(output_pattern).exists() and Path(output_pattern).is_file():
        overwrite = Prompt.ask(f"{output_pattern} already exists, do you "
                               f"want to overwrite it?",
                               choices=["y", "n"],
                               default="y")
        if overwrite == "n":
            output_pattern = Prompt.ask(f"Please enter another output "
                                        f"filename",
                                        default=".\\outputs\\new_output")
            output_pattern = handle_output_pattern(output_pattern, ext)
        else:
            break

    total_batches = (len(image_files) + n - 1) // n

    if total_batches == 1:
        concatenate_images_vertically(image_files, output_pattern, width)
    else:
        if "[]" not in output_pattern:
            output_pattern = output_pattern.replace(f".{ext}", f"[].{ext}")
        zero_padded_width = len(str(total_batches))
        for batch_idx in range(total_batches):
            batch_files = image_files[batch_idx * n:(batch_idx + 1) * n]
            if not batch_files:
                continue
            output_path = output_pattern.replace(
                "[]", f"{batch_idx + 1:0{zero_padded_width}}")
            concatenate_images_vertically(batch_files, output_path, width)


def expand_patterns(patterns, ext):
    expanded_list = []
    for pattern in patterns:
        match = re.search(r'\[(\d+)-(\d+)]', pattern)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            num_length = len(match.group(1))
            for i in range(start, end + 1):
                expanded_path = re.sub(r'\[\d+-\d+]', f"{i:0{num_length}}",
                                       pattern)
                expanded_list.append(expanded_path)
        else:
            match_single = re.search(r'\[(\d)-(\d)]', pattern)
            if match_single:
                start, end = int(match_single.group(1)), int(
                    match_single.group(2))
                for i in range(start, end + 1):
                    expanded_path = re.sub(r'\[\d-\d]', f"{i}", pattern)
                    expanded_list.append(expanded_path)
            else:
                expanded_list.append(pattern)

    unfolded_list = []
    for expanded_path in expanded_list:
        path = Path(expanded_path)
        if not path.exists():
            logger.error(f"File or directory not exists - {path}")
        elif path.is_file():
            unfolded_list.append(path)
        elif path.is_dir():
            files = sorted(path.glob(f"*.{ext}"))
            unfolded_list.extend(files)

    return unfolded_list


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate images vertically.")
    parser.add_argument("input_paths", nargs="*",
                        help="Input image paths or patterns")
    parser.add_argument("-o", "--output_path",
                        default=".\\outputs\\output.jpg",
                        help="Output image path or pattern")
    parser.add_argument("-n", type=int, default=6,
                        help="Number of images per batch")
    parser.add_argument("-e", "--extension", default="jpg",
                        help="File extension for images")
    parser.add_argument("-w", "--width", type=int,
                        help="Width of output images")
    return parser.parse_args()


args = parse_args()
input_paths = expand_patterns(args.input_paths, args.extension)
if not input_paths:
    logger.error(
        "No valid image files found based on the input.")
else:
    batch_concatenate_images(input_paths, args.output_path, args.n,
                             args.extension, args.width)
