import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert('RGB')
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))

    # Prepare an image
    image = load_image(args.image, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        if word == '<start>':
            continue
        if word == '<end>':
            break
        sampled_caption.append(word)
    sentence = ' '.join(sampled_caption).capitalize()
    
    # Print out the image and the generated caption
    print(f"Generated Caption: {sentence}")
    image = Image.open(args.image)
    plt.imshow(np.asarray(image))
    plt.title(sentence)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # --- THESE LINES ARE NOW FIXED (used add_argument) ---
    parser.add_argument('--image', type=str, required=True, help='input image for caption generation')
    parser.add_argument('--encoder_path', type=str, default='models/encoder.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder.pkl', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should match the trained model)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    
    if not os.path.exists(args.image):
        print(f"Error: Image file not found at {args.image}")
    elif not os.path.exists(args.encoder_path):
        print(f"Error: Encoder file not found at {args.encoder_path}")
    elif not os.path.exists(args.decoder_path):
        print(f"Error: Decoder file not found at {args.decoder_path}")
    elif not os.path.exists(args.vocab_path):
        print(f"Error: Vocabulary file not found at {args.vocab_path}")
    else:
        main(args)