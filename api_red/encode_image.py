import os
import json
import base64
import argparse
from tqdm import tqdm

def encode_image(image_path):
    """
    Encodes an image into a base64 string.
    :param image_path: Path to the input image.
    :return: Base64 encoded string of the image.
    """
    try:
        with open(image_path, "rb") as image_file:
            # Read image and encode it to base64
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        return encoded_image
    except FileNotFoundError:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    except Exception as e:
        raise RuntimeError(f"Error encoding image {image_path}: {e}")

def save_encoded_image(encoded_data, output_path):
    """
    Saves the encoded image data to a JSON file.
    :param encoded_data: Encoded image data as a dictionary.
    :param output_path: Path to save the JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(encoded_data, f, indent=4)

def process_images(image_dir, output_dir):
    """
    Encodes all images in a directory and saves the results to JSON files.
    :param image_dir: Directory containing images to encode.
    :param output_dir: Directory to save the encoded JSON files.
    """
    if not os.path.exists(image_dir):
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    if not os.path.isdir(image_dir):
        raise ValueError(f"Provided image path is not a directory: {image_dir}")

    images = os.listdir(image_dir)
    for img in tqdm(images):
        img_path = os.path.join(image_dir, img)
        if os.path.isfile(img_path):  # Ensure it's a file
            encoded_image = encode_image(img_path)
            output_path = os.path.join(output_dir, f"{img}.json")
            save_encoded_image({"encoded_image": encoded_image}, output_path)
    print(f"Images encoded and saved: {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode images in a directory to JSON files with base64 encoding.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the directory containing images.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the directory to save encoded JSON files.")

    args = parser.parse_args()

    try:
        process_images(args.image_dir, args.output_dir)
    except Exception as e:
        print(f"Error: {e}")
