# Personal Image Generation with DALL-E

Fine-tune a lightweight, open-source DALL-E model on your personal photos to generate new AI images featuring you in any setting or style you can imagine! This project allows you to create a personalized image generation model with just a few photos.

> **⚠️ Disclaimer**  
> Training generative models on personal images can potentially reveal identity information.  
> Please keep your training data private and use generated images responsibly.

## Setup

1. **Environment Setup**
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

2. **Prepare Training Data**
   - Create a `training_images` directory
   - Add 5-20 high-quality photos of the subject
   - Photos should be:
     - Well-lit and clear
     - Showing different angles/expressions
     - In JPG or PNG format
     - Ideally with neutral backgrounds

## CLI Usage

The project provides a user-friendly command-line interface with two main commands: `train` and `generate`.

### Training the Model

```bash
python cli.py train \
    --subject_token "your_name" \
    --epochs 5 \
    --batch_size 4 \
    --lr 1e-4
```

Training Parameters:
- `--subject_token` (required): Unique identifier for the person (e.g., "john" or "sarah")
- `--image_dir`: Directory containing training images (default: "training_images")
- `--epochs`: Number of training iterations (default: 5)
- `--batch_size`: Images per training batch (default: 4)
- `--lr`: Learning rate (default: 0.0001)
- `--resume_path`: Path to checkpoint to resume training from (optional)
- `--save_every`: Save checkpoint every N epochs (default: 1)

### Generating Images

```bash
python cli.py generate \
    --prompt "a photo of your_name in paris" \
    --dalle_ckpt "checkpoints/dalle.pt" \
    --num_images 4
```

Generation Parameters:
- `--prompt` (required): Text description including your subject_token
- `--dalle_ckpt` (required): Path to your trained model checkpoint
- `--num_images`: Number of images to generate (default: 4)
- `--temperature`: Controls randomness (0.0-1.0, default: 1.0)
- `--top_k`: Controls diversity (higher = more diverse, default: 64)
- `--seed`: Random seed for reproducibility (optional)

### Example Workflow

1. **Train the model**:
```bash
# Basic training
python cli.py train --subject_token "john"

# Advanced training with custom parameters
python cli.py train \
    --subject_token "john" \
    --epochs 10 \
    --batch_size 8 \
    --lr 5e-5 \
    --save_every 2
```

2. **Generate images**:
```bash
# Basic generation
python cli.py generate \
    --prompt "a photo of john as an astronaut" \
    --dalle_ckpt "checkpoints/dalle.pt"

# Advanced generation with custom parameters
python cli.py generate \
    --prompt "a photo of john in a renaissance painting" \
    --dalle_ckpt "checkpoints/dalle.pt" \
    --num_images 6 \
    --temperature 0.8 \
    --top_k 128 \
    --seed 42
```

## Tips for Best Results

1. **Training Data Quality**
   - Use consistent image sizes
   - Avoid heavily edited or filtered photos
   - Include a variety of facial expressions
   - Ensure good lighting and clear visibility

2. **Prompt Engineering**
   - Always include "a photo of [subject_token]" in your prompts
   - Be specific about the setting, style, or context
   - Examples:
     - "a photo of john as an astronaut"
     - "a photo of sarah in a renaissance painting"
     - "a professional headshot of alex"

## Technical Details

This implementation uses:
- DALL-E pytorch implementation by lucidrains
- OpenAI's discrete VAE architecture
- Lightweight transformer configuration for quick fine-tuning
- CUDA acceleration when available

## Requirements

- Python 3.11+
- PyTorch with CUDA (recommended) or CPU
- See requirements.txt for full dependencies

## License

This project is MIT licensed. Please respect privacy and data protection laws when using this tool.

---

## 1. Quick-start

```bash
# 1. Clone your project
git clone https://github.com/<you>/personal-dalle.git
cd personal-dalle

# 2. Python env & deps  (CUDA-enabled wheels recommended)
python -m venv .venv && source .venv/bin/activate
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu118
pip install dalle-pytorch==1.10.2 pillow tqdm

# 3. Add 5–20 neutral, well-lit photos of ONE person to:
mkdir training_images
# ➜  copy *.jpg or *.png here
