from flask import Flask, request, jsonify, send_file
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont
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

# Color configuration dictionary
COLOR_DICT = {
    "red": [255, 0, 0, 255],
    "green": [0, 255, 0, 255],
    "blue": [0, 0, 255, 255],
    "white": [255, 255, 255, 255],
    "black": [0, 0, 0, 255],
    "yellow": [255, 255, 0, 255],
    "purple": [128, 0, 128, 255],
    "orange": [255, 165, 0, 255],
    "pink": [255, 192, 203, 255],
    "brown": [165, 42, 42, 255],
    "gray": [128, 128, 128, 255]
}

def save_blob_to_image(file, save_dir="uploads"):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, file.filename)
    file.save(file_path)
    return file_path

def predict_step(images):
    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds

def change_bg_color(img, color):
    output = remove(img)
    image = output.convert("RGBA")
    data = np.array(image)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i, j][3] == 0:
                data[i, j] = color
    return Image.fromarray(data)

def convert_color_to_rgb(color):
    """Convert color name or hex to RGB tuple"""
    if color.startswith('#'):
        # Convert hex to RGB
        color = color.lstrip('#')
        return tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
    else:
        # Use predefined colors or default to black
        color_values = COLOR_DICT.get(color.lower(), COLOR_DICT["black"])
        return tuple(color_values[:3])  # Return only RGB values, exclude alpha

def add_text_to_image(img, text1="", text2="", font_size1=100, font_size2=20, position="center", 
                      text1_color="black", text2_color="black"):
    """
    Add text to the image with improved rendering using transparent layer
    """
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Load custom font
    font_path = "./fonts/DejaVuSans.ttf"
    try:
        font1 = ImageFont.truetype(font_path, font_size1)
        font2 = ImageFont.truetype(font_path, font_size2)
    except IOError:
        print("Font not found. Using default font.")
        font1 = ImageFont.load_default()
        font2 = ImageFont.load_default()

    # Calculate positions
    positions = calculate_text_position(img, text1, text2, font1, font2, position.lower())

    # Convert colors to RGB
    color1 = convert_color_to_rgb(text1_color)
    color2 = convert_color_to_rgb(text2_color)

    # Create transparent text overlay layer
    text_layer = Image.new("RGBA", img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(text_layer)

    # Draw text onto overlay
    if text1:
        draw.text(positions["text1"], text1, font=font1, fill=color1)
    if text2:
        draw.text(positions["text2"], text2, font=font2, fill=color2)

    # Composite the overlay onto the original image
    img = Image.alpha_composite(img, text_layer)

    return img

def calculate_text_position(img, text1, text2, font1, font2, position):
    """Calculate text positions based on the specified position parameter"""
    draw = ImageDraw.Draw(img)
    
    text1_bbox = draw.textbbox((0, 0), text1, font=font1)
    text1_width = text1_bbox[2] - text1_bbox[0]
    text1_height = text1_bbox[3] - text1_bbox[1]
    
    text2_bbox = draw.textbbox((0, 0), text2, font=font2)
    text2_width = text2_bbox[2] - text2_bbox[0]
    text2_height = text2_bbox[3] - text2_bbox[1]
    
    total_height = text1_height + text2_height + 15
    padding = 20
    
    positions = {
        "top": {
            "text1": ((img.width - text1_width) // 2, padding),
            "text2": ((img.width - text2_width) // 2, padding + text1_height + 15)
        },
        "bottom": {
            "text1": ((img.width - text1_width) // 2, img.height - total_height - padding),
            "text2": ((img.width - text2_width) // 2, img.height - text2_height - padding)
        },
        "center": {
            "text1": ((img.width - text1_width) // 2, (img.height - total_height) // 2),
            "text2": ((img.width - text2_width) // 2, (img.height - total_height) // 2 + text1_height + 15)
        },
        "left": {
            "text1": (padding, (img.height - total_height) // 2),
            "text2": (padding, (img.height - total_height) // 2 + text1_height + 15)
        },
        "right": {
            "text1": (img.width - text1_width - padding, (img.height - total_height) // 2),
            "text2": (img.width - text2_width - padding, (img.height - total_height) // 2 + text1_height + 15)
        }
    }
    
    return positions.get(position, positions["center"])

@app.route('/generate-logo', methods=['POST'])
def generate_logo():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    image_file = request.files['image']
    image_path = save_blob_to_image(image_file)
    try:
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        caption = predict_step([image])
        prompt = f"Generate a Icon of {caption[0]} in middle, no text, flat, minimalist."
        generated_image = pipe(prompt).images[0]
        img_io = io.BytesIO()
        generated_image.save(img_io, format="PNG")
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": f"Failed to generate logo: {str(e)}"}), 500

@app.route('/edit-logo', methods=['POST'])
def edit_logo_api():
    if 'logo' not in request.files:
        return jsonify({"error": "No logo file provided"}), 400

    logo_file = request.files['logo']
    logo_path = save_blob_to_image(logo_file)

    # Get parameters from the request
    name = request.form.get('name', '')
    slogan = request.form.get('slogan', '')
    background = request.form.get('background', '')
    name_font_size = int(request.form.get('name_font_size', 50))
    text_position = request.form.get('position', 'center')
    name_color = request.form.get('name_color', 'black')  # New parameter for name color
    slogan_color = request.form.get('slogan_color', 'black')  # New parameter for slogan color

    # Validate position parameter
    valid_positions = ['top', 'right', 'left', 'center', 'bottom']
    if text_position.lower() not in valid_positions:
        return jsonify({"error": f"Invalid position. Must be one of: {', '.join(valid_positions)}"}), 400

    try:
        img = Image.open(logo_path)

        if background and background.strip().lower() != "select color if you want":
            img = change_bg_color(img, COLOR_DICT.get(background.lower(), [255, 255, 255, 255]))

        if name or slogan:
            img = add_text_to_image(
                img,
                text1=name,
                text2=slogan,
                font_size1=name_font_size,
                font_size2=20,
                position=text_position,
                text1_color=name_color,
                text2_color=slogan_color
            )

        img_io = io.BytesIO()
        # img.save(img_io, format="PNG")
        img.save(img_io, format="PNG", dpi=(300, 300))
        img_io.seek(0)
        return send_file(img_io, mimetype="image/png")
    except Exception as e:
        return jsonify({"error": f"Failed to edit logo: {str(e)}"}), 500
    
@app.route('/generate-logo-text', methods=['POST'])
def generate_logo_text():
    caption = request.form.get('caption')
    if caption is None:
        return jsonify({"error": "Caption path provided"}), 400
    
    # Generate caption from image

    # Generate logo using Stable Diffusion
    prompt = f"Generate a Icon of {caption} in middle, no text, flat, minimalist."
    image = pipe(prompt).images[0]

    # Save the generated logo
    # logo_path = "results/generated_logo_text.png"
    # image.save(logo_path)

    # return jsonify({"message": "Logo generated successfully", "path": logo_path}), 200
    img_io = io.BytesIO()
    image.save(img_io, format="PNG")
    img_io.seek(0)
    return send_file(img_io, mimetype="image/png")


if __name__ == '__main__':
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning", cache_dir="./model_cache")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
    max_length = 16
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    app.run(debug=True, port=5000)

