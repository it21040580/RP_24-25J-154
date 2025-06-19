

### API Endpoints for Logo Editing and Generation

This Flask application provides APIs for editing logos and generating new ones using image captioning and Stable Diffusion.

---

### **1. `/edit-logo`**

**Description**: Edits an uploaded logo by changing its background color and adding text (name and slogan).

#### **Request Type**: `POST`

#### **Parameters**:
- **Form Data**:
  - `logo` (string) - The logo image to be edited (required).
  - `name` (string) - The name text to be added to the logo (optional).
  - `slogan` (string) - The slogan text to be added to the logo (optional).
  - `background` (string) - Background color (e.g., "red", "blue", "green", "white", "black") (optional).

#### **Response**:
- **Success**: 
  ```json
  {
    "message": "Logo edited successfully",
    "path": "edited/edited_logo.png"
  }
  ```
- **Error**:
  ```json
  {
    "error": "No logo file provided"
  }
  ```
---

### **2. `/generate-logo`**

**Description**: Generates a new logo based on the content of an uploaded image. The process includes:
1. Generating a caption for the image.
2. Using the caption as input to Stable Diffusion to create a minimalist logo.

#### **Request Type**: `POST`

#### **Parameters**:
- **Form Data**:
  - `image` (string) - The input image path used to generate the logo (required).

#### **Response**:
- **Success**: 
  ```json
  {
    "message": "Logo generated successfully",
    "path": "results/generated_logo.png"
  }
  ```
- **Error**:
  ```json
  {
    "error": "No image file provided"
  }
  ```

---

### **Usage Examples**

#### **Generating a Logo**
```bash
curl -X POST -F "image=@/path/to/input_image.jpg" \
http://127.0.0.1:5000/generate-logo
```
 
#### **Editing a Logo**
```bash
curl -X POST -F "logo=@/path/to/logo.png" \
-F "name=My Brand" -F "slogan=Innovation First" \
-F "background=blue" \
http://127.0.0.1:5000/edit-logo
```