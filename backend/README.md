# Personal DALL-E Backend

Flask API server for fine-tuning and running a lightweight DALL-E model on personal photos.

## Architecture

The backend consists of several key components:

- `app.py`: Flask RESTful API with endpoints for uploads, training, and generation
- `dalle_person.py`: Core functionality for DALL-E model training and image generation
- `cli.py`: Command-line interface for the same functionality

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/status` | GET | Check server status and CUDA availability |
| `/upload` | POST | Upload training images for a subject |
| `/train` | POST | Train the model on a subject's images |
| `/generate` | POST | Generate images using a trained model |
| `/checkpoints` | GET | List available model checkpoints |
| `/subjects` | GET | List available subject tokens |
| `/images/<filename>` | GET | Serve generated images |
| `/training-images/<subject>` | GET | List training images for a subject |
| `/training-images/<subject>/<filename>` | GET | Serve training images |

## Requirements

- Python 3.11+
- PyTorch (>=2.2.0)
- CUDA-enabled GPU (highly recommended)
- Required packages in `requirements.txt`

## Installation

1. Create and activate a Python virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. (Optional) To use CUDA acceleration, ensure you have compatible NVIDIA drivers installed.

## Running the Server

Start the Flask API server:

```bash
python app.py
```

The server will start on http://localhost:5000 by default.

## Using the Command Line Interface

In addition to the API, you can use the CLI for direct interactions:

### Training a Model

```bash
python cli.py train --subject_token "john" --epochs 5
```

Optional parameters:
- `--image_dir`: Directory containing training images (default: "training_images")
- `--epochs`: Number of training iterations (default: 5)
- `--batch_size`: Images per batch (default: 4)
- `--lr`: Learning rate (default: 1e-4)
- `--save_every`: Save checkpoint frequency (default: 1)

### Generating Images

```bash
python cli.py generate --prompt "a photo of john in paris" --dalle_ckpt "checkpoints/dalle.pt"
```

Optional parameters:
- `--num_images`: Number of images to generate (default: 4)
- `--temperature`: Controls randomness (0.0-1.5, default: 1.0)
- `--top_k`: Controls diversity (default: 64)
- `--seed`: Random seed for reproducibility

## Technical Notes

- The model uses a modified version of lucidrains' DALL-E implementation
- Training creates a checkpoint file in the `checkpoints/` directory
- Generated images are saved in `generated_images/`
- All images are stored as PNG files

## Troubleshooting

Common issues:

1. **CUDA out of memory**: Reduce batch size or try running on CPU.
2. **Missing dependencies**: Ensure all requirements are installed.
3. **Permission errors**: Check filesystem permissions for the directories.
4. **Slow performance**: Train with fewer epochs or use a GPU.

## Development

To modify the API or add new features:

1. Modify `app.py` to add new endpoints
2. Update `dalle_person.py` for model architecture changes
3. Test locally before deploying

## License

This project is MIT licensed.

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
