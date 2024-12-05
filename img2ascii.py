import argparse

from PIL import Image, UnidentifiedImageError


def generate_ascii_from_image(image_path, max_width=160, max_height=90,
                              output_file=None):
    try:
        with Image.open(image_path) as img:
            original_width, original_height = img.size
            print(
                f"Successfully loaded image! "
                f"Original image size: {original_width} x {original_height}")

            img.thumbnail((max_width, max_height))
            width, height = img.size
            print(f"Resized image size: {width} x {height}")

            pixel_data = [[img.getpixel((x, y)) for x in range(width)] for y in
                          range(height)]
            print(f"Successfully constructed pixel data! "
                  f"Pixel data size: {width} x {height}")

            luminosity_values = [
                [int(0.21 * pixel[0] + 0.72 * pixel[1] + 0.07 * pixel[2]) for
                 pixel in row] for row in pixel_data]
            print(f"Successfully constructed luminosity values! "
                  f"Luminosity values size: {width} x {height}")

            characters = ("`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvcz"
                          "XYUJCLQ0OZmwqpdbkhao*#MW&8%B@$")

            ascii_art = []
            for row in luminosity_values:
                ascii_row = []
                for luminosity in row:
                    index = round((luminosity / 255) * (len(characters) - 1))
                    ascii_row.append(characters[index])
                ascii_art.append(ascii_row)

            print(f"Successfully constructed ASCII art! "
                  f"ASCII art size: {width} x {height}")

            ascii_output = "\n".join(
                ["".join([char * 2 for char in row]) for row in ascii_art])

            if output_file:
                with open(output_file, "w") as f:
                    f.write(ascii_output)
                print(f"ASCII art has been saved to {output_file}")
            else:
                print(ascii_output)

    except FileNotFoundError:
        print(f"Error: The specified image file '{image_path}' was not found.")
    except UnidentifiedImageError:
        print(f"Error: The file '{image_path}' is not a valid image.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert image to ASCII art and optionally save to a file.")
    parser.add_argument("input_paths", nargs="+", help="Input image paths")
    parser.add_argument("-W", "--width", type=int, default=160,
                        help="Max width of output images (default: 160)")
    parser.add_argument("-H", "--height", type=int, default=90,
                        help="Max height of output images (default: 90)")
    parser.add_argument("-o", "--output", type=str,
                        help="Output ASCII art to a text file")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print(f"Using the following parameters:")
    print(f"Max width: {args.width}")
    print(f"Max height: {args.height}")

    if args.output:
        print(f"Output will be saved to: {args.output}")

    for path in args.input_paths:
        generate_ascii_from_image(path, args.width, args.height,
                                  args.output)
