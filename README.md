# Personal DALL-E Image Generation

Fine-tune a lightweight DALL-E model on your personal photos to generate AI images featuring you in any setting or style. With just a few photos, you can create a personalized image generation model that puts you in scenarios limited only by your imagination.

## Overview

This project consists of:
- **Backend API**: Flask server that handles image processing, model training, and image generation
- **Frontend UI**: React web interface for uploading photos, training models, and generating images

## Features

- **Upload Training Images**: Upload 5-20 photos of yourself or any subject
- **Train Custom Models**: Fine-tune a DALL-E model with your photos
- **Generate Images**: Create AI images with customizable prompts and parameters

## Project Structure

```
ImageGen/
├── backend/             # Flask API server
│   ├── app.py           # Main Flask application
│   ├── dalle_person.py  # Core DALL-E functionality
│   ├── cli.py           # Command line interface
│   └── requirements.txt # Python dependencies
│
├── frontend/            # React web application
│   ├── src/             # React source code
│   ├── public/          # Static assets
│   └── package.json     # Node.js dependencies
│
├── training_images/     # Created during runtime for storing training images
└── generated_images/    # Created during runtime for storing generated images
```

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- PyTorch with CUDA (recommended) or CPU

### 1. Set Up Backend

```bash
# Navigate to backend directory
cd backend

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start the Flask server
python3 app.py
```

The backend server will start on http://localhost:5001

### 2. Set Up Frontend

```bash
# In a new terminal, navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend will start on http://localhost:5173

### 3. Use the Application

1. Open your browser and navigate to http://localhost:5173
2. Follow the workflow in the UI:
   - Upload images
   - Train model
   - Generate images

## Detailed Documentation

For detailed setup and usage instructions:
- [Backend Documentation](backend/README.md)
- [Frontend Documentation](frontend/README.md)

## Performance Notes

- Training works best with 5-20 clear, well-lit photos showing different angles/expressions
- GPU is highly recommended for faster training and generation
- Training typically takes 5-15 minutes depending on hardware and parameters
- Generation takes 5-20 seconds per image

## License

This project is MIT licensed. Please respect privacy and data protection laws when using this tool.

## Acknowledgements

- DALL-E pytorch implementation by [lucidrains](https://github.com/lucidrains/DALLE-pytorch)
- OpenAI's discrete VAE architecture
- React for the frontend UI 