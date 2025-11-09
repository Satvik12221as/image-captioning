# 🤖 Image Captioning (LLaVA 7B)

I have configured this repository to run a multimodal model, **LLaVA (Large Language and Vision Assistant)**.

This is a powerful visual analysis tool. It uses the LLaVA model's deep, internet-scale knowledge to generate captions that are far more accurate, detailed, and context-aware.

---

## Key Features

* **State-of-the-Art Model:** Implements `llava-hf/llava-1.5-7b-hf`, a 7-billion parameter model for deep visual understanding.
* **Deep Contextual Knowledge:** The model's vast training allows it to recognize specific people, characters, and landmarks with high accuracy.
* **GPU-Optimized:** The code uses `BitsAndBytesConfig` for 4-bit quantization, a technique that compresses the massive model to run on free or consumer-grade GPUs (like the T4 in Google Colab).

---

## 💻 How to Run This Project (Recommended: Google Colab)

This is a 7B parameter model. It is **massive** (over 10GB in memory) and **requires a powerful GPU** to run. The easiest and most accessible way to run this is using a **free T4 GPU** provided by Google Colab.

### Step 1: Open Colab & Enable GPU

1.  Go to [colab.research.google.com](https://colab.research.google.com) and start a **New Notebook**.
2.  In the menu, go to **Runtime - Change runtime type**.
3.  Select **T4 GPU**.

### Step 2: Clone Project and run the script

Run this in a single Colab cell. It will clone your repository, move into the project folder, and install all necessary libraries.

```
!git clone [https://github.com/Satvik12221as/image-captioning.git](https://github.com/Satvik12221as/image-captioning.git)
%cd image-captioning
!pip install -r requirements.txt
!pip install bitsandbytes accelerate
!python llava_model.py
from IPython.display import Image
Image('images/image1.jpg', width=400) # Displaying the image with a set width
```

---

## Project Demo

![alt text](<Screenshot 2025-11-09 221343.png>)