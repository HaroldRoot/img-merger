import argparse
from collections import Counter
from pathlib import Path

import cv2
import imagehash
import numpy as np
import pytesseract
from PIL import Image


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


def locate_subtitle(path, output_path, debug=False):
    # Load the image
    img_data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

    height, width, _ = img.shape

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform OCR using Tesseract
    custom_config = r'--oem 3 --psm 6 -l chi_sim'  # Use Chinese language pack
    data = pytesseract.image_to_data(gray, config=custom_config,
                                     output_type=pytesseract.Output.DICT)

    # Output debug information
    if debug:
        print("OCR Detection Results:")
        for i in range(len(data['text'])):
            if data['text'][i].strip():
                print(f"Text: {data['text'][i]}, "
                      f"Position: ({data['left'][i]}, {data['top'][i]}) -> "
                      f"({data['width'][i]}, {data['height'][i]})")

    # Collect y-coordinates and heights of all valid text boxes
    y_positions = []
    heights = []
    for i in range(len(data['text'])):
        text = data['text'][i].strip()
        if text:  # If valid text is detected
            y_positions.append(int(data['top'][i]))
            heights.append(int(data['height'][i]))

    # If no valid text boxes are found, attempt edge-based detection
    if not y_positions:
        if debug:
            print("No OCR-based subtitle candidates found. "
                  "Attempting edge-based detection.")
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        min_y = height
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            # Assume subtitles are horizontal strips
            if y > height * 0.5 and 3 < aspect_ratio < 100:
                min_y = min(min_y, y)
        if min_y < height:
            result = img[min_y:height, 0:width]
            cv2.imwrite(output_path, result)
            print(f"Subtitle region saved to {output_path} "
                  f"(fallback detection, start_y={min_y}).")
        else:
            print("No subtitle found.")
        return

    # Determine the cropping region
    # based on the mode of y-coordinates and heights
    y_counter = Counter(y_positions)
    height_counter = Counter(heights)

    start_y = max(y_counter, key=y_counter.get)  # Mode of y-coordinates
    subtitle_height = max(height_counter,
                          key=height_counter.get)  # Mode of heights

    # Prevent cropping beyond image bounds
    start_y = max(0, start_y)
    # Allow slight extension for multi-line subtitles
    end_y = min(height, start_y + subtitle_height * 2)

    # Crop the subtitle region
    result = img[start_y:end_y, 0:width]
    cv2.imwrite(output_path, result)
    print(f"Subtitle region saved to {Path(output_path).resolve()} "
          f"(start_y={start_y}, end_y={end_y}).")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate screenshots vertically.")
    parser.add_argument('input_paths', nargs='*',
                        help="Input image paths or patterns")
    parser.add_argument('-o', '--output_path', default='output.jpg',
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
    parser.add_argument('-v', '--debug', '--verbose', action='store_true',
                        help='Enable debug mode for detailed logs.')
    return parser.parse_args()


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
    locate_subtitle(args.locsub, output_path=args.output_path, debug=args.debug)
