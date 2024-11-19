import argparse
from pathlib import Path

import imagehash
from PIL import Image


def calculate_image_similarity(path1, path2):
    image1 = Image.open(path1)
    image2 = Image.open(path2)

    hash1 = imagehash.average_hash(image1)
    hash2 = imagehash.average_hash(image2)

    result = 1 - (hash1 - hash2) / len(hash1.hash) ** 2

    return result


def get_images_between(from_path, to_path):
    from_path_obj = Path(from_path)
    to_path_obj = Path(to_path)
    dir_path = from_path_obj.parent
    images = sorted([p for p in dir_path.iterdir() if
                     p.is_file() and p.suffix in ['.jpg', '.png']])
    start_index = images.index(from_path_obj)
    end_index = images.index(to_path_obj)
    return [str(p) for p in images[start_index:end_index + 1]]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Concatenate screenshots vertically.")
    parser.add_argument('input_paths', nargs='*',
                        help="Input image paths or patterns")
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
    return parser.parse_args()


args = parse_args()

if args.compare:
    image_path1 = Path(args.compare[0])
    image_path2 = Path(args.compare[1])
    similarity = calculate_image_similarity(image_path1, image_path2)
    print(f"The similarity of the two images is: {similarity}")
elif args.cluster:
    clusters = []
    reference_image = None
    threshold = args.threshold

    if args.from_path and args.to_path:
        input_paths = get_images_between(args.from_path, args.to_path)
    else:
        input_paths = args.input_paths

    for image_path in input_paths:
        if reference_image is None:
            reference_image = image_path
            clusters.append([reference_image])
            continue

        similarity = calculate_image_similarity(Path(reference_image),
                                                Path(image_path))
        if similarity >= threshold:
            clusters[-1].append(image_path)
        else:
            reference_image = image_path
            clusters.append([reference_image])

    print(clusters)
