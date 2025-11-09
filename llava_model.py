import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, BitsAndBytesConfig
import os
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Define Model and Image ---
model_id = "llava-hf/llava-1.5-7b-hf"
image_path = "images/image2.jpg"

# --- 2. Configure 4-bit Loading ---
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading model... This will take a while...")
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto"
)
print("Model loaded.")

# --- 4. Open the Image ---
try:
    raw_image = Image.open(image_path).convert('RGB')
except FileNotFoundError:
    print(f"Error: Image file not found at {image_path}")
    exit()

# --- 5. Create the Prompt ---
prompt = f"USER: <image>\nWho are these characters? Be specific. ASSISTANT:"

print("Processing image and prompt...")
inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(model.device)

# --- 6. Generate the Response ---
print("Generating response...")
output = model.generate(**inputs, max_new_tokens=100)
response_full = processor.decode(output[0], skip_special_tokens=True)

try:
    response_clean = response_full.split("ASSISTANT:")[1].strip()
except IndexError:
    response_clean = "Model did not generate a valid response."

# --- 7. Display Both the Text and the Image ---

# 7a. Print the text (for you to copy)
print("\n--- LLaVA Model Response ---")
print(response_clean)

# 7b. Show the image "pop-up" (inline)
print("\nShowing image with caption...")
plt.figure(figsize=(10, 12))  # Make figure larger to fit text
plt.imshow(np.asarray(raw_image))
plt.title(response_clean, fontsize=10, pad=20, wrap=True) # Add caption as title
plt.axis('off')
plt.show() # This displays the plot in Colab