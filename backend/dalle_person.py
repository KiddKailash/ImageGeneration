#!/usr/bin/env python3
"""
Fine-tune an open-source DALL-E model on photos of a single person,
then generate new images featuring that person.

Author: 2025-05-06
"""

import argparse
import glob
import os
from pathlib import Path
from typing import List

import torch
from PIL import Image
from tqdm import tqdm

# ---- DALLE-pytorch (lucidrains) imports -------------------------------------
#   pip install dalle-pytorch==1.10.2
from dalle_pytorch import DiscreteVAE, DALLE
from dalle_pytorch.tokenizer import SimpleTokenizer
from dalle_pytorch.vae import OpenAIDiscreteVAE
from dalle_pytorch.dataloaders import TextImageDataset, ImageDataset
from dalle_pytorch.trainer import DALLETrainer

# --------------------------------------------------------------------------- #

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GENERATED_DIR = Path("generated_images")
GENERATED_DIR.mkdir(exist_ok=True)
TOKENIZER = SimpleTokenizer()

# --------------------------------------------------------------------------- #
#                               TRAINING
# --------------------------------------------------------------------------- #

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

    # DALLE-pytorch expects a TSV captions file in the folder
    dataset = TextImageDataset(
        text_file      = str(captions_path),
        image_folder   = str(image_dir),
        tokenizer      = TOKENIZER,
        image_size     = image_size,
        truncate_captions = True,
    )
    return dataset


def train(
    image_dir: str,
    subject_token: str,
    epochs: int,
    batch_size: int,
    lr: float,
    resume_path: str | None,
    save_every: int = 1,
):
    image_dir = Path(image_dir)
    dataset   = build_training_dataset(image_dir, subject_token)

    vae = OpenAIDiscreteVAE().to(DEVICE)        # loads pretrained VAE weights
    dalle = DALLE(
        dim               = 512,
        vae               = vae,
        num_text_tokens   = TOKENIZER.vocab_size,
        text_seq_len      = 128,
        depth             = 2,   # extremely shallow – good enough for few-shot tuning
        heads             = 8,
        dim_head          = 64,
    ).to(DEVICE)

    trainer = DALLETrainer(
        dalle             = dalle,
        dall_e_path       = resume_path,
        dataloader        = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2),
        lr                = lr,
        epochs            = epochs,
        save_every        = save_every,
        generate_every    = 0,
        fp16              = torch.cuda.is_available(),
    )

    trainer.train()


# --------------------------------------------------------------------------- #
#                               GENERATION
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate(prompt: str, dalle_checkpoint: str, num_images: int, top_k: int, temperature: float, seed: int | None):
    if seed is not None:
        torch.manual_seed(seed)

    vae      = OpenAIDiscreteVAE().to(DEVICE)
    dalle    = DALLE.load(dalle_checkpoint, vae=vae, device=DEVICE)  # Trainer's .pt file
    dalle.eval()

    tokens   = TOKENIZER.tokenize(prompt).unsqueeze(0).to(DEVICE)
    images   = dalle.generate_images(tokens, filter_thres=top_k, temperature=temperature, cond_scale=3.)

    for idx, img in enumerate(images):
        pil = Image.fromarray(img.cpu().numpy())
        out_path = GENERATED_DIR / f"{prompt.replace(' ','_')[:50]}_{idx:02d}.png"
        pil.save(out_path)
        print(f"✅  Saved {out_path}")


# --------------------------------------------------------------------------- #
#                               CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune open-source DALL-E on photos of ONE person, then generate novel images."
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # ---- train -------------------------------------------------------------- #
    t = subparsers.add_parser("train", help="fine-tune on images in training_images/")
    t.add_argument("--image_dir",      type=str, default="training_images", help="folder of photos")
    t.add_argument("--subject_token",  type=str, required=True, help="unique token representing the person, e.g. 'kailash'")
    t.add_argument("--epochs",         type=int, default=5)
    t.add_argument("--batch_size",     type=int, default=4)
    t.add_argument("--lr",             type=float, default=1e-4)
    t.add_argument("--resume_path",    type=str, default=None, help="path to previous .pt checkpoint")
    t.add_argument("--save_every",     type=int, default=1, help="save checkpoint every N epochs")

    # ---- generate ----------------------------------------------------------- #
    g = subparsers.add_parser("generate", help="create images after training")
    g.add_argument("--prompt",         type=str, required=True, help="text prompt containing the subject token")
    g.add_argument("--dalle_ckpt",     type=str, required=True, help="path to trained DALLE .pt checkpoint")
    g.add_argument("--num_images",     type=int, default=4)
    g.add_argument("--top_k",          type=int, default=64)
    g.add_argument("--temperature",    type=float, default=1.0)
    g.add_argument("--seed",           type=int, default=None)

    args = parser.parse_args()

    if args.cmd == "train":
        train(
            image_dir     = args.image_dir,
            subject_token = args.subject_token,
            epochs        = args.epochs,
            batch_size    = args.batch_size,
            lr            = args.lr,
            resume_path   = args.resume_path,
            save_every    = args.save_every,
        )
    elif args.cmd == "generate":
        generate(
            prompt        = args.prompt,
            dalle_checkpoint = args.dalle_ckpt,
            num_images    = args.num_images,
            top_k         = args.top_k,
            temperature   = args.temperature,
            seed          = args.seed,
        )


if __name__ == "__main__":
    main()
