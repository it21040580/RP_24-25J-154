from flask import Flask, request, jsonify, send_file
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
from diffusers import StableDiffusionPipeline
from rembg import remove
import numpy as np
import os
import io
import multiprocessing
from flask_cors import CORS

try:
    multiprocessing.set_start_method("fork")
except RuntimeError:
    pass

app = Flask(__name__)
CORS(app)  # Enable CORS for all origins

MOCK_IMAGE_PATH = "results/generated-logo (1).png"  # Path to the mock image

# Function to save blob image
def save_blob_to_image(file, save_dir="uploads"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, file.filename)
    file.save(file_path)
    return file_path

def change_bg_color(img, color):
    output = remove(img)
    image = output.convert("RGBA")
    data = np.array(image)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j][3] == 0:
                data[i, j] = color
    return Image.fromarray(data)

def add_text_to_image(img, text1="", text2="", font_size1=100, font_size2=20):
    # Ensure image is in a compatible mode
    if img.mode != "RGBA":
        img = img.convert("RGBA")
    
    # Specify the font path
    font_path = "./fonts/DejaVuSans.ttf"  # Replace with a valid font file path on your system

    # Load the fonts
    try:
        from PIL import ImageFont, ImageDraw
        font1 = ImageFont.truetype(font_path, font_size1)
        font2 = ImageFont.truetype(font_path, font_size2)
    except IOError:
        print("Font not found. Using a default font.")
        font1 = ImageFont.load_default()
        font2 = ImageFont.load_default()

    draw = ImageDraw.Draw(img)

    # Get bounding boxes for both texts
    text1_bbox = draw.textbbox((0, 0), text1, font=font1)
    text1_width, text1_height = text1_bbox[2] - text1_bbox[0], text1_bbox[3] - text1_bbox[1]

    text2_bbox = draw.textbbox((0, 0), text2, font=font2)
    text2_width, text2_height = text2_bbox[2] - text2_bbox[0], text2_bbox[3] - text2_bbox[1]

    # Calculate total height of both texts to center them vertically
    total_height = text1_height + text2_height
    vertical_center = (img.height - total_height) - 75

    # Calculate the position for the first and second text
    text1_position = ((img.width - text1_width) // 2, vertical_center)
    text2_position = ((img.width - text2_width) // 2, vertical_center + text1_height + 15)

    # Draw the texts on the image
    draw.text(text1_position, text1, font=font1, fill="black")  # Add first text in black color
    draw.text(text2_position, text2, font=font2, fill="black")  # Add second text in black color

    return img


@app.route('/generate-logo', methods=['POST'])
def generate_logo():
    # Mock response or logic for the `generate-logo` endpoint
    if not os.path.exists(MOCK_IMAGE_PATH):
        return jsonify({"error": f"Mock image not found at {MOCK_IMAGE_PATH}"}), 500
    try:
        return send_file(MOCK_IMAGE_PATH, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": f"Failed to send mock image: {str(e)}"}), 500


@app.route('/generate-logo-text', methods=['POST'])
def generate_logo_text():
    """
    Returns a mock image for testing instead of generating a new one.
    """
    caption = request.form.get('caption')
    if not caption:
        return jsonify({"error": "Caption not provided"}), 400

    if not os.path.exists(MOCK_IMAGE_PATH):
        return jsonify({"error": f"Mock image not found at {MOCK_IMAGE_PATH}"}), 500

    try:
        return send_file(MOCK_IMAGE_PATH, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": f"Failed to send mock image: {str(e)}"}), 500

@app.route('/edit-logo', methods=['POST'])
def edit_logo_api():
    """
    Handles editing of logos by adding text and background changes.
    """
    if 'logo' not in request.files:
        return jsonify({"error": "No logo file provided"}), 400

    logo_file = request.files['logo']
    logo_path = save_blob_to_image(logo_file)

    name = request.form.get('name', '')
    slogan = request.form.get('slogan', '')
    background = request.form.get('background', '')
    name_font_size = int(request.form.get('name_font_size', 50))

    try:
        img = Image.open(logo_path)
        if background:
            color_dict = {
                "red": [255, 0, 0, 255],
                "green": [0, 255, 0, 255],
                "blue": [0, 0, 255, 255],
                "white": [255, 255, 255, 255],
                "black": [0, 0, 0, 255],
            }
            img = change_bg_color(img, color_dict.get(background, [255, 255, 255, 255]))

        if name or slogan:
            img = add_text_to_image(img, text1=name, text2=slogan, font_size1=name_font_size, font_size2=20)

        img_io = io.BytesIO()
        img.save(img_io, format="PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": f"Failed to edit logo: {str(e)}"}), 500

if __name__ == '__main__':
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir="./model_cache")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    app.run(debug=True, port=5000)
