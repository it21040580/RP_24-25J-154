import streamlit as st
import torch
import pickle
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn

# Define device (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load saved generator model
class Generator(nn.Module):
    def __init__(self, text_embedding_size=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(100 + text_embedding_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 49152),
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        combined_input = torch.cat((noise, text_embedding), dim=1)
        return self.model(combined_input)

# Load the trained generator model and vectorizer
generator = Generator(text_embedding_size=100).to(device)
generator.load_state_dict(torch.load('generator.pth', weights_only=True))
generator.eval()

# Load vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Define a function to generate images based on input text
def generate_image_from_text(description):
    # Convert text description to vector
    desc_vector = torch.tensor(vectorizer.transform([description]).toarray(), dtype=torch.float32).to(device)
    
    # Generate random noise
    noise = torch.randn(1, 100).to(device)
    
    # Generate image using the generator model
    with torch.no_grad():
        generated_image = generator(noise, desc_vector).view(3, 128, 128)
        generated_image = (generated_image + 1) / 2  # Rescale to [0, 1]
    
    return generated_image

# Streamlit app interface
st.title("Image Generator from Text Description")

# Text input for description
description = st.text_input("Enter a description for the image:")

# Generate and display the image
if st.button("Generate Image"):
    if description:
        st.write(f"Generating image for: {description}")

        # Generate the image based on the description
        generated_image = generate_image_from_text(description)

        # Convert to numpy array for display
        generated_image = generated_image.cpu().numpy().transpose(1, 2, 0)

        # Show the image using Streamlit
        st.image(generated_image, caption="Generated Image", use_container_width=True)
    else:
        st.write("Please enter a valid description.")
