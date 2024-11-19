import argparse
import re
from pathlib import Path

from PIL import Image
from rich.prompt import Prompt, Confirm

from custom_logger import logger


def concatenate_images_vertically(image_paths, output_path, width):
    images = []
    for image_path in image_paths:
        try:
            img = Image.open(image_path)
            images.append(img)
        except FileNotFoundError:
            logger.error(f"File not found - {image_path}")
        except IOError:
            logger.error(f"Could not open image file - {image_path}")

    if not images:
        logger.error("No valid images to concatenate.")
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
                                     Image.Resampling.BICUBIC)

        # Paste the resized image onto the new_image canvas
        new_image.paste(resized_image, (0, y_offset))
        y_offset += resized_height

    try:
        new_image.save(output_path)
        abs_output_path = Path(output_path).resolve()
        logger.info(f"Concatenated image saved to {abs_output_path}")
    except IOError:
        logger.error("Could not save the concatenated image.")
    except KeyError as e:
        logger.error(f"{e} is not a valid extension for output files.")
    except ValueError as e:
        if str(e) != "unknown file extension: ":
            logger.error("U" + str(e)[1:])
        logger.error("You must specify a valid file extension for the -o "
                     "parameter, such as output.jpg")


def handle_output_pattern(pattern):
    if "\\" in pattern:
        output_dir = Path(pattern).parent
        output_dir.mkdir(parents=True, exist_ok=True)
    return pattern


def add_brackets_to_output_pattern(output_pattern):
    last_dot_index = output_pattern.rfind('.')
    if last_dot_index != -1:
        return output_pattern[:last_dot_index] + '[]' + output_pattern[
                                                        last_dot_index:]
    else:
        return output_pattern + '[]'


def batch_concatenate_images(image_files, output_pattern, n, width):
    output_pattern = handle_output_pattern(output_pattern)
    while Path(output_pattern).exists() and Path(output_pattern).is_file():
        overwrite = Confirm.ask(f"{output_pattern} already exists, do you "
                                f"want to overwrite it?",
                                default=True)
        if not overwrite:
            output_pattern = Prompt.ask(f"Please enter another output "
                                        f"filename",
                                        default=".\\outputs\\new_output.jpg")
            output_pattern = handle_output_pattern(output_pattern)
        else:
            break

    total_batches = (len(image_files) + n - 1) // n

    if total_batches == 1:
        concatenate_images_vertically(image_files, output_pattern, width)
    else:
        if "[]" not in output_pattern:
            output_pattern = add_brackets_to_output_pattern(output_pattern)
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
            files = []
            for ext_part in ext.split(','):
                files.extend(sorted(path.glob(f"*.{ext_part.strip()}")))
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
    parser.add_argument("-e", "--extension", default="jpg,jpeg,png",
                        help="File extension for input images")
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
                             args.width)
