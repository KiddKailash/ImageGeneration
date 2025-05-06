#!/usr/bin/env python3
"""
Flask API for Personal DALL-E Image Generation.
"""

import os
import sys
import uuid
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch

from dalle_model import train, generate, build_training_dataset

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Constants
UPLOAD_FOLDER = Path("training_images")
GENERATED_FOLDER = Path("generated_images")
CHECKPOINT_DIR = Path("checkpoints")
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png"}

# Create necessary directories
UPLOAD_FOLDER.mkdir(exist_ok=True)
GENERATED_FOLDER.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)

@app.route('/status', methods=['GET'])
def status():
    """Check if server is running."""
    return jsonify({
        "status": "online",
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    })

@app.route('/upload', methods=['POST'])
def upload_images():
    """
    Upload training images.
    Expected format: multipart/form-data with 'images' files and 'subject_token' field.
    """
    if 'images' not in request.files:
        return jsonify({"error": "No images provided"}), 400
    
    subject_token = request.form.get('subject_token')
    if not subject_token:
        return jsonify({"error": "No subject token provided"}), 400
    
    # Create subject-specific directory
    subject_dir = UPLOAD_FOLDER / subject_token
    subject_dir.mkdir(exist_ok=True)
    
    files = request.files.getlist('images')
    saved_files = []
    
    for file in files:
        if file.filename:
            # Sanitize filename
            filename = f"{uuid.uuid4()}{Path(file.filename).suffix.lower()}"
            if Path(filename).suffix.lower() not in SUPPORTED_FORMATS:
                continue
                
            file_path = subject_dir / filename
            file.save(file_path)
            saved_files.append(str(file_path))
    
    if not saved_files:
        return jsonify({"error": "No valid images uploaded"}), 400
    
    return jsonify({
        "message": f"Successfully uploaded {len(saved_files)} images for {subject_token}",
        "subject_token": subject_token,
        "image_count": len(saved_files)
    })

@app.route('/train', methods=['POST'])
def train_model():
    """
    Train the model on uploaded images.
    Expected JSON: {
        "subject_token": "string",
        "epochs": int,
        "batch_size": int,
        "lr": float,
        "resume_path": "string" (optional)
    }
    """
    data = request.json
    
    if not data or 'subject_token' not in data:
        return jsonify({"error": "Missing parameters"}), 400
    
    subject_token = data['subject_token']
    image_dir = UPLOAD_FOLDER / subject_token
    
    if not image_dir.exists() or not any(image_dir.glob("*")):
        return jsonify({"error": f"No images found for subject token: {subject_token}"}), 404
    
    # Extract parameters with defaults
    epochs = data.get('epochs', 5)
    batch_size = data.get('batch_size', 4)
    lr = data.get('lr', 1e-4)
    resume_path = data.get('resume_path')
    save_every = data.get('save_every', 1)
    
    # Start training in a separate process (for real implementation)
    # Here we'll just call the function directly, but in production you'd use a task queue
    try:
        checkpoint_path = train(
            image_dir=str(image_dir),
            subject_token=subject_token,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            resume_path=resume_path,
            save_every=save_every
        )
        
        return jsonify({
            "message": "Training completed successfully",
            "subject_token": subject_token,
            "checkpoint": str(checkpoint_path)
        })
    except Exception as e:
        return jsonify({
            "error": f"Training failed: {str(e)}"
        }), 500

@app.route('/generate', methods=['POST'])
def generate_images():
    """
    Generate images using the trained model.
    Expected JSON: {
        "prompt": "string",
        "dalle_checkpoint": "string",
        "num_images": int,
        "top_k": int,
        "temperature": float,
        "seed": int (optional)
    }
    """
    data = request.json
    
    if not data or 'prompt' not in data or 'dalle_checkpoint' not in data:
        return jsonify({"error": "Missing parameters"}), 400
    
    prompt = data['prompt']
    dalle_checkpoint = data['dalle_checkpoint']
    
    # Check if checkpoint exists
    checkpoint_path = Path(dalle_checkpoint)
    if not checkpoint_path.exists():
        return jsonify({"error": f"Checkpoint not found: {dalle_checkpoint}"}), 404
    
    # Extract parameters with defaults
    num_images = data.get('num_images', 4)
    top_k = data.get('top_k', 64)
    temperature = data.get('temperature', 1.0)
    seed = data.get('seed')
    
    try:
        generated_paths = generate(
            prompt=prompt,
            dalle_checkpoint=str(checkpoint_path),
            num_images=num_images,
            top_k=top_k,
            temperature=temperature,
            seed=seed
        )
        
        # Return paths to generated images
        image_urls = [f"/images/{path.name}" for path in generated_paths]
        
        return jsonify({
            "message": "Images generated successfully",
            "prompt": prompt,
            "images": image_urls
        })
    except Exception as e:
        return jsonify({
            "error": f"Generation failed: {str(e)}"
        }), 500

@app.route('/checkpoints', methods=['GET'])
def list_checkpoints():
    """List available model checkpoints."""
    checkpoints = [f.name for f in CHECKPOINT_DIR.glob("*.pt")]
    return jsonify({
        "checkpoints": checkpoints
    })

@app.route('/subjects', methods=['GET'])
def list_subjects():
    """List available subject tokens (folders in training_images)."""
    subjects = [d.name for d in UPLOAD_FOLDER.iterdir() if d.is_dir()]
    return jsonify({
        "subjects": subjects
    })

@app.route('/images/<path:filename>', methods=['GET'])
def get_image(filename):
    """Serve generated images."""
    return send_from_directory(GENERATED_FOLDER, filename)

@app.route('/training-images/<subject>/<path:filename>', methods=['GET'])
def get_training_image(subject, filename):
    """Serve training images."""
    return send_from_directory(UPLOAD_FOLDER / subject, filename)

@app.route('/training-images/<subject>', methods=['GET'])
def list_training_images(subject):
    """List training images for a specific subject."""
    subject_dir = UPLOAD_FOLDER / subject
    if not subject_dir.exists():
        return jsonify({"error": f"Subject not found: {subject}"}), 404
        
    images = [f.name for f in subject_dir.glob("*") if f.suffix.lower() in SUPPORTED_FORMATS]
    image_urls = [f"/training-images/{subject}/{img}" for img in images]
    
    return jsonify({
        "subject": subject,
        "images": image_urls,
        "count": len(image_urls)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True) 