#!/usr/bin/env python3
"""
Fine-tune an open-source DALL-E model on photos of a single person,
then generate new images featuring that person.
"""

import argparse
import os
import numpy as np
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch.optim import Adam
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset

# Import dalle-pytorch components that we know exist
from dalle_pytorch import DiscreteVAE, DALLE
from dalle_pytorch.tokenizer import SimpleTokenizer
from dalle_pytorch.vae import OpenAIDiscreteVAE

# CONSTANTS
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATED_DIR = Path("generated_images")
CHECKPOINT_DIR = Path("checkpoints")
GENERATED_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
TOKENIZER = SimpleTokenizer()

# Custom dataset implementations to avoid dependency issues
class TextImageDataset(Dataset):
    def __init__(self, text_file, image_folder, tokenizer, image_size=256, truncate_captions=False):
        self.text_file = text_file
        self.image_folder = image_folder
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.truncate_captions = truncate_captions
        
        # Read captions file
        with open(text_file, 'r') as f:
            self.captions = []
            self.image_paths = []
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                image_filename, caption = parts[0], parts[1]
                image_path = os.path.join(image_folder, image_filename)
                if os.path.exists(image_path):
                    self.captions.append(caption)
                    self.image_paths.append(image_path)
    
    def __len__(self):
        return len(self.captions)
    
    def __getitem__(self, idx):
        caption = self.captions[idx]
        image_path = self.image_paths[idx]
        
        # Process image
        image = Image.open(image_path).convert('RGB')
        
        # Center crop and resize
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        image = image.crop((left, top, right, bottom))
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(
            np.array(image).astype(np.float32) / 127.5 - 1
        ).permute(2, 0, 1)
        
        # Process caption
        if self.truncate_captions:
            caption = caption[:76]
        caption_tokens = self.tokenizer.tokenize(caption)
        
        return caption_tokens, image_tensor

class ImageDataset(Dataset):
    def __init__(self, folder, image_size=256):
        self.folder = folder
        self.image_size = image_size
        self.image_files = [
            f for f in os.listdir(folder)
            if f.endswith(('.jpg', '.jpeg', '.png'))
        ]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.folder, self.image_files[idx])
        
        # Process image
        image = Image.open(image_path).convert('RGB')
        
        # Center crop and resize
        width, height = image.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        image = image.crop((left, top, right, bottom))
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(
            np.array(image).astype(np.float32) / 127.5 - 1
        ).permute(2, 0, 1)
        
        return image_tensor

# Custom implementation of training without the trainer module
def train_one_epoch(dalle, optimizer, dataloader, device, epoch):
    dalle.train()
    total_loss = 0
    
    with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}") as pbar:
        for batch_idx, (text, images) in enumerate(dataloader):
            text, images = text.to(device), images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            loss = dalle(text, images, return_loss=True)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            pbar.set_postfix(loss=avg_loss)
            pbar.update()
    
    return total_loss / len(dataloader)

# Training and generation functions
def build_training_dataset(image_dir: Path, subject_token: str, image_size: int = 256):
    """
    Build a tiny <caption, image> dataset by auto-generating captions.
    Every caption is simply "a photo of {subject_token}".

    Args:
        image_dir: directory containing JPG / PNG photos
        subject_token: unique name you will use inside prompts
        image_size: images are center-cropped & resized to this
    """
    captions_path = image_dir / "captions.txt"
    with captions_path.open("w") as f:
        for img_path in sorted(image_dir.glob("*")):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            f.write(f"{img_path.name}\ta photo of {subject_token}\n")

    # Create dataset with our custom implementation
    dataset = TextImageDataset(
        text_file=str(captions_path),
        image_folder=str(image_dir),
        tokenizer=TOKENIZER,
        image_size=image_size,
        truncate_captions=True,
    )
    return dataset


def train(
    image_dir: str,
    subject_token: str,
    epochs: int,
    batch_size: int,
    lr: float,
    resume_path: Optional[str] = None,
    save_every: int = 1,
):
    """
    Train DALL-E model on personal images.
    
    Returns:
        Path to the saved checkpoint file
    """
    image_dir = Path(image_dir)
    dataset = build_training_dataset(image_dir, subject_token)

    vae = OpenAIDiscreteVAE().to(DEVICE)  # loads pretrained VAE weights
    dalle = DALLE(
        dim=512,
        vae=vae,
        num_text_tokens=TOKENIZER.vocab_size,
        text_seq_len=128,
        depth=2,  # extremely shallow – good enough for few-shot tuning
        heads=8,
        dim_head=64,
    ).to(DEVICE)
    
    # Load checkpoint if provided
    if resume_path is not None:
        dalle.load_state_dict(torch.load(resume_path))
        print(f"Loaded checkpoint from {resume_path}")

    # Create a checkpoint path with unique name
    final_checkpoint_path = CHECKPOINT_DIR / f"dalle_{subject_token}_{epochs}ep.pt"
    
    # Create optimizer
    optimizer = Adam(dalle.parameters(), lr=lr)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(epochs):
        avg_loss = train_one_epoch(dalle, optimizer, dataloader, DEVICE, epoch)
        print(f"Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
        
        if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
            # Save checkpoint
            save_path = CHECKPOINT_DIR / f"dalle_{subject_token}_epoch{epoch+1}.pt"
            torch.save(dalle.state_dict(), save_path)
            print(f"Saved checkpoint to {save_path}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(dalle.state_dict(), final_checkpoint_path)
                print(f"Saved best model to {final_checkpoint_path}")
    
    return final_checkpoint_path


@torch.no_grad()
def generate(prompt: str, dalle_checkpoint: str, num_images: int, top_k: int, temperature: float, seed: Optional[int] = None):
    """
    Generate images using the trained model.
    
    Returns:
        List of paths to the generated images
    """
    if seed is not None:
        torch.manual_seed(seed)

    vae = OpenAIDiscreteVAE().to(DEVICE)
    
    # Load the model
    dalle = DALLE(
        dim=512,
        vae=vae,
        num_text_tokens=TOKENIZER.vocab_size,
        text_seq_len=128,
        depth=2,
        heads=8,
        dim_head=64,
    ).to(DEVICE)
    
    dalle.load_state_dict(torch.load(dalle_checkpoint))
    dalle.eval()

    # Tokenize the prompt
    text = TOKENIZER.tokenize([prompt]).to(DEVICE)
    
    # Generate images
    images = []
    for _ in range(num_images):
        img = dalle.generate_images(
            text,
            filter_thres=top_k / 100.0,
            temperature=temperature
        )
        images.append(img[0])  # Take the first (and only) image

    # Save the generated images
    generated_paths = []
    for idx, img in enumerate(images):
        # Convert to PIL and save
        pil_img = Image.fromarray(img.cpu().numpy())
        out_path = GENERATED_DIR / f"{prompt.replace(' ', '_')[:50]}_{idx:02d}.png"
        pil_img.save(out_path)
        print(f"✅ Saved {out_path}")
        generated_paths.append(out_path)
    
    return generated_paths


# CLI functionality
def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune open-source DALL-E on photos of ONE person, then generate novel images."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # Train command
    t = subparsers.add_parser("train", help="fine-tune on images in training_images/")
    t.add_argument("--image_dir", type=str, default="training_images", help="folder of photos")
    t.add_argument("--subject_token", type=str, required=True, help="unique token representing the person, e.g. 'kailash'")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--batch_size", type=int, default=4)
    t.add_argument("--lr", type=float, default=1e-4)
    t.add_argument("--resume_path", type=str, default=None, help="path to previous .pt checkpoint")
    t.add_argument("--save_every", type=int, default=1, help="save checkpoint every N epochs")

    # Generate command
    g = subparsers.add_parser("generate", help="create images after training")
    g.add_argument("--prompt", type=str, required=True, help="text prompt containing the subject token")
    g.add_argument("--dalle_ckpt", type=str, required=True, help="path to trained DALLE .pt checkpoint")
    g.add_argument("--num_images", type=int, default=4)
    g.add_argument("--top_k", type=int, default=64)
    g.add_argument("--temperature", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.cmd == "train":
        checkpoint_path = train(
            image_dir=args.image_dir,
            subject_token=args.subject_token,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            resume_path=args.resume_path,
            save_every=args.save_every,
        )
        print(f"Training completed. Final checkpoint saved at: {checkpoint_path}")
    elif args.cmd == "generate":
        generated_paths = generate(
            prompt=args.prompt,
            dalle_checkpoint=args.dalle_ckpt,
            num_images=args.num_images,
            top_k=args.top_k,
            temperature=args.temperature,
            seed=args.seed,
        )
        print(f"Generated {len(generated_paths)} images in {GENERATED_DIR}")


if __name__ == "__main__":
    main()