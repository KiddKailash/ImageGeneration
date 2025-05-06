#!/usr/bin/env python3
"""
Command Line Interface for Personal DALL-E Image Generation
Handles training and generation workflows with rich feedback and error handling.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List, Optional
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.logging import RichHandler
from rich import print as rprint
import torch

from dalle_model import train, generate

# Configure rich console for better output
console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("rich")

# Constants
DEFAULT_IMAGE_DIR = "training_images"
DEFAULT_OUTPUT_DIR = "generated_images"
CHECKPOINT_DIR = Path("checkpoints")
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}

def setup_directories() -> None:
    """Create necessary directories if they don't exist."""
    for directory in [DEFAULT_IMAGE_DIR, DEFAULT_OUTPUT_DIR, CHECKPOINT_DIR]:
        Path(directory).mkdir(exist_ok=True)

def validate_training_images(image_dir: Path) -> List[Path]:
    """
    Validate that training images exist and are in correct format.
    Returns list of valid image paths.
    """
    if not image_dir.exists():
        console.print(f"[red]Error: Training directory '{image_dir}' does not exist!")
        sys.exit(1)
        
    images = [
        p for p in image_dir.iterdir() 
        if p.suffix.lower() in SUPPORTED_FORMATS
    ]
    
    if not images:
        console.print(f"[red]Error: No supported images found in '{image_dir}'")
        console.print(f"[yellow]Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        sys.exit(1)
        
    return images

def validate_checkpoint(checkpoint_path: Optional[str]) -> Optional[Path]:
    """Validate that checkpoint file exists if specified."""
    if checkpoint_path:
        path = Path(checkpoint_path)
        if not path.exists():
            console.print(f"[red]Error: Checkpoint file '{path}' not found!")
            sys.exit(1)
        return path
    return None

def train_command(args: argparse.Namespace) -> None:
    """Handle the training workflow."""
    console.print("[bold blue]Starting training workflow...[/]")
    
    # Validate inputs
    image_dir = Path(args.image_dir)
    images = validate_training_images(image_dir)
    console.print(f"[green]Found {len(images)} valid training images[/]")
    
    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[blue]Using device: [bold]{device}[/]")
    
    if device == "cpu":
        console.print("[yellow]Warning: Training on CPU will be significantly slower!")
        
    # Start training
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Training model...", total=args.epochs)
            
            def progress_callback(epoch: int):
                progress.update(task, advance=1)
            
            train(
                image_dir=str(image_dir),
                subject_token=args.subject_token,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                resume_path=args.resume_path,
                save_every=args.save_every,
            )
            
        console.print("[bold green]Training completed successfully![/]")
        
    except Exception as e:
        console.print(f"[red]Error during training: {str(e)}[/]")
        sys.exit(1)

def generate_command(args: argparse.Namespace) -> None:
    """Handle the image generation workflow."""
    console.print("[bold blue]Starting image generation...[/]")
    
    # Validate checkpoint
    checkpoint = validate_checkpoint(args.dalle_ckpt)
    if not checkpoint:
        console.print("[red]Error: No checkpoint specified!")
        sys.exit(1)
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating images...")
            
            generate(
                prompt=args.prompt,
                dalle_checkpoint=str(checkpoint),
                num_images=args.num_images,
                top_k=args.top_k,
                temperature=args.temperature,
                seed=args.seed
            )
            
            progress.update(task, completed=True)
            
        console.print(f"[bold green]Successfully generated {args.num_images} images![/]")
        console.print(f"[blue]Images saved in: {DEFAULT_OUTPUT_DIR}/")
        
    except Exception as e:
        console.print(f"[red]Error during generation: {str(e)}[/]")
        sys.exit(1)

def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Personal DALL-E Image Generation CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Training command
    train_parser = subparsers.add_parser(
        "train",
        help="Train model on personal images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    train_parser.add_argument(
        "--image_dir",
        type=str,
        default=DEFAULT_IMAGE_DIR,
        help="Directory containing training images"
    )
    train_parser.add_argument(
        "--subject_token",
        type=str,
        required=True,
        help="Unique identifier for the subject (e.g., 'john')"
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Number of training epochs"
    )
    train_parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Training batch size"
    )
    train_parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate"
    )
    train_parser.add_argument(
        "--resume_path",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    train_parser.add_argument(
        "--save_every",
        type=int,
        default=1,
        help="Save checkpoint every N epochs"
    )
    
    # Generation command
    gen_parser = subparsers.add_parser(
        "generate",
        help="Generate new images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    gen_parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt for image generation"
    )
    gen_parser.add_argument(
        "--dalle_ckpt",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    gen_parser.add_argument(
        "--num_images",
        type=int,
        default=4,
        help="Number of images to generate"
    )
    gen_parser.add_argument(
        "--top_k",
        type=int,
        default=64,
        help="Top k filtering for generation"
    )
    gen_parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0.0-1.0)"
    )
    gen_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    
    # Parse and execute
    args = parser.parse_args()
    setup_directories()
    
    try:
        if args.command == "train":
            train_command(args)
        elif args.command == "generate":
            generate_command(args)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/]")
        sys.exit(0)

if __name__ == "__main__":
    main() 