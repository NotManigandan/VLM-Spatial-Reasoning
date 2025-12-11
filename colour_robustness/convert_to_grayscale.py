import os
from PIL import Image
import argparse

def convert_to_grayscale(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, filename)
            img = Image.open(img_path).convert('L')  # Convert to grayscale
            img.save(os.path.join(output_folder, filename))
            print(f"Converted {filename} to grayscale.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to grayscale")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the input image folder")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to the output grayscale image folder")
    args = parser.parse_args()

    convert_to_grayscale(args.input_folder, args.output_folder)