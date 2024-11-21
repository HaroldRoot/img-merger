import argparse
from collections import Counter
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pytesseract
from PIL import Image

from img_stitcher import handle_output_pattern


def calculate_image_similarity(paths):
    image1 = Image.open(Path(paths[0]))
    image2 = Image.open(Path(paths[1]))

    hash1 = imagehash.average_hash(image1)
    hash2 = imagehash.average_hash(image2)

    result = 1 - (hash1 - hash2) / len(hash1.hash) ** 2
    return result


def cluster_images(image_paths, threshold):
    clusters = []
    reference_image = None

    for image_path in image_paths:
        if reference_image is None:
            reference_image = image_path
            clusters.append([reference_image])
            continue

        similarity = calculate_image_similarity(
            [Path(reference_image), Path(image_path)])
        if similarity >= threshold:
            clusters[-1].append(image_path)
        else:
            reference_image = image_path
            clusters.append([reference_image])

    return clusters


def get_images_between(from_path, to_path):
    from_path_obj = Path(from_path)
    to_path_obj = Path(to_path)
    dir_path = from_path_obj.parent
    images = sorted([p for p in dir_path.iterdir() if
                     p.is_file() and p.suffix in ['.jpg', '.png']])
    start_index = images.index(from_path_obj)
    end_index = images.index(to_path_obj)
    return [str(p) for p in images[start_index:end_index + 1]]


def locate_subtitle(path, output_path='.\\outputs\\output.jpg', debug=False):
    def save_debug_image(image, filename, description=""):
        filepath = Path(output_path).parent / filename
        cv2.imwrite(filepath, image)
        print(f"{description} saved to: {Path(filepath).resolve()}")

    # Load the image
    img_data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    height, width, _ = img.shape

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Directly convert to binary image using thresholding
    _, binary_image = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    if debug:
        save_debug_image(binary_image, "1_binary_image.jpg",
                         "Step 1: Binary image")

    # Perform OCR using Tesseract
    custom_config = r'--oem 3 --psm 6 -l chi_sim'  # Use Chinese language pack
    data = pytesseract.image_to_data(binary_image, config=custom_config,
                                     output_type=pytesseract.Output.DICT)

    # Output debug information
    if debug:
        print("OCR Detection Results:")
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                print(f"Text: {data['text'][i]}, "
                      f"Position: ({data['left'][i]}, {data['top'][i]}) -> "
                      f"({data['width'][i]}, {data['height'][i]})")

        # Mark OCR detected text regions on the original image
        debug_image = img.copy()
        for i in range(len(data['text'])):
            if data['text'][i].strip():  # Valid text
                x, y, w, h = (data['left'][i], data['top'][i], data['width'][i],
                              data['height'][i])
                cv2.rectangle(debug_image, (x, y), (x + w, y + h), (0, 0, 255),
                              2)  # Red rectangle
        save_debug_image(debug_image, "2_ocr_detected_regions.jpg",
                         "Step 2: OCR detected text regions")

    # Collect y-coordinates and heights of all valid text boxes
    y_positions = []
    heights = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:  # If valid text is detected
            y_positions.append(int(data['top'][i]))
            heights.append(int(data['height'][i]))

    # Determine the cropping region
    y_counter = Counter(y_positions)
    height_counter = Counter(heights)

    start_y = max(y_counter, key=y_counter.get)  # Mode of y-coordinates
    subtitle_height = max(height_counter,
                          key=height_counter.get)  # Mode of heights

    # Prevent cropping beyond image bounds
    start_y = max(0, start_y)
    # Allow slight extension for multi-line subtitles
    end_y = min(height, start_y + subtitle_height * 2)

    if debug:
        # Mark cropping region on the original image
        debug_crop_image = img.copy()
        cv2.rectangle(debug_crop_image, (0, start_y), (width, end_y),
                      (0, 255, 0), 2)  # Green rectangle
        save_debug_image(debug_crop_image, "3_final_subtitle_region.jpg",
                         "Step 3: Final subtitle cropping region")

        # Crop the subtitle region
        result = img[start_y:end_y, 0:width]
        save_debug_image(result, "4_final_image.jpg", "Step 4: Final image")

    return start_y, end_y


def concatenate_screenshots(image_paths, output_path, threshold, debug=False):
    output_path = handle_output_pattern(output_path)
    clusters = cluster_images(image_paths, threshold)
    results = []

    for i, cluster in enumerate(clusters):
        if len(cluster) == 1:
            # If there is only one image in the cluster, save it directly
            img = cv2.imdecode(np.fromfile(cluster[0], dtype=np.uint8),
                               cv2.IMREAD_COLOR)
            output_file = Path(output_path).parent / f"cluster_{i + 1}.jpg"
            cv2.imwrite(str(output_file), img)
            results.append(str(output_file))
            continue

        # Get the full image region from the first image in the cluster
        base_image_path = cluster[0]
        base_image = cv2.imdecode(np.fromfile(base_image_path, dtype=np.uint8),
                                  cv2.IMREAD_COLOR)

        # Extract and stack subtitle regions from other images in the cluster
        subtitle_images = []
        for img_path in cluster[1:]:
            subtitle_coords = locate_subtitle(img_path, debug=debug)
            if subtitle_coords:
                start_y, end_y = subtitle_coords
                subtitle_img = cv2.imdecode(
                    np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                subtitle_images.append(subtitle_img[start_y:end_y, :])

        # Combine the main image with the subtitle regions
        result_img = base_image
        for sub_img in subtitle_images:
            result_img = np.vstack((result_img, sub_img))

        # Save the result
        output_file = Path(output_path).parent / f"cluster_{i + 1}.jpg"
        cv2.imwrite(str(output_file), result_img)
        results.append(str(output_file))

    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate screenshots vertically.")
    parser.add_argument('input_paths', nargs='*',
                        help="Input image paths or patterns")
    parser.add_argument('-o', '--output_path', default='.\\outputs\\output.jpg',
                        help="Output image path or pattern")
    parser.add_argument('--compare', nargs=2,
                        metavar=('image_path1', 'image_path2'),
                        help='Compare the similarity of two images, and the '
                             'paths of the two images need to be provided.')
    parser.add_argument('--cluster', action='store_true',
                        help='Enable clustering mode.')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Similarity threshold for clustering. '
                             'Default is 0.9.')
    parser.add_argument('--from', dest='from_path',
                        help='Starting path for selecting images in a '
                             'directory.')
    parser.add_argument('--to', dest='to_path',
                        help='Ending path for selecting images in a directory.')
    parser.add_argument('--locsub',
                        help='Locate and extract the subtitle part of the '
                             'image.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode.')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.from_path and args.to_path:
        input_paths = get_images_between(args.from_path, args.to_path)
    else:
        input_paths = args.input_paths

    if args.compare:
        print(f"The similarity of the two images is: "
              f"{calculate_image_similarity(args.compare)}")
    elif args.cluster:
        print(cluster_images(input_paths, threshold=args.threshold))
    elif args.locsub:
        locate_subtitle(args.locsub, output_path=args.output_path, debug=True)
    elif not input_paths:
        print("No valid image files found based on the input.")
    else:
        concatenate_screenshots(input_paths, args.output_path, args.threshold)
