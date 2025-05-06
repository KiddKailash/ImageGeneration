# Personal DALL-E Frontend

React web application for interacting with the DALL-E personal image generation API.

## Features

- **Image Upload**: Upload training photos with a simple drag-and-drop interface
- **Model Training**: Configure and initiate model training through the UI
- **Image Generation**: Generate new AI images with customizable parameters
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

- Built with React 19+
- Uses Axios for API communication
- Styled with TailwindCSS
- Vite for fast development and building

## Project Structure

```
frontend/
├── src/                 # React source code
│   ├── components/      # React components
│   │   ├── Upload.jsx   # Upload interface component
│   │   ├── Training.jsx # Training interface component
│   │   └── Generation.jsx # Generation interface component
│   ├── App.jsx          # Main application component
│   ├── main.jsx         # Application entry point
│   └── index.css        # Global CSS including Tailwind
├── public/              # Static assets
├── index.html           # HTML entry point
├── package.json         # Dependencies and scripts
├── vite.config.js       # Vite configuration
├── tailwind.config.js   # Tailwind CSS configuration
└── postcss.config.js    # PostCSS configuration
```

## Requirements

- Node.js 18+
- npm or yarn
- Backend API server running (see backend README)

## Installation

1. Install dependencies:

```bash
npm install
```

2. Configure environment if needed:

For custom API URL, create a `.env` file in the frontend directory:

```
VITE_API_URL=http://your-backend-url:5000
```

## Development

Start the development server:

```bash
npm run dev
```

The application will be available at http://localhost:5173 by default.

## Building for Production

Build the application for production:

```bash
npm run build
```

This generates optimized files in the `dist/` directory.

## Previewing the Production Build

To preview the production build locally:

```bash
npm run preview
```

## Component Overview

### Upload Component

Handles image uploading:
- Subject token input
- File selection
- Image previews
- Upload progress

### Training Component

Controls model training:
- Subject selection
- Training parameters (epochs, batch size, learning rate)
- Training progress
- Training results

### Generation Component

Provides image generation:
- Model checkpoint selection
- Prompt input with suggestions
- Generation parameters (temperature, top-k sampling)
- Generated image display and download

## Customization

### Styling

The application uses TailwindCSS. To customize styles:

1. Modify component classes
2. Update `tailwind.config.js` for theme changes
3. Add custom styles in `index.css`

### API Integration

API communication is handled through Axios. The base URL is set in `App.jsx`:

```jsx
const API_URL = 'http://localhost:5000';
```

Update this value to match your backend server address.

## Troubleshooting

Common issues:

1. **API Connection Errors**: Check if backend server is running and CORS is enabled
2. **Image Upload Failures**: Verify file types (only JPG/PNG supported) and network connectivity
3. **UI Rendering Issues**: Check browser console for errors and component props

## License

This project is MIT licensed.
