import { useState } from 'react';

// MUI components
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Slider from '@mui/material/Slider';
import FormHelperText from '@mui/material/FormHelperText';
import CircularProgress from '@mui/material/CircularProgress';
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';
import Divider from '@mui/material/Divider';
import Card from '@mui/material/Card';
import CardMedia from '@mui/material/CardMedia';
import CardContent from '@mui/material/CardContent';
import CardActions from '@mui/material/CardActions';
import IconButton from '@mui/material/IconButton';
import Tooltip from '@mui/material/Tooltip';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Alert from '@mui/material/Alert';

// MUI icons
import Image from '@mui/icons-material/Image';
import ContentCopy from '@mui/icons-material/ContentCopy';
import Add from '@mui/icons-material/Add';
import Delete from '@mui/icons-material/Delete';
import Download from '@mui/icons-material/Download';
import Refresh from '@mui/icons-material/Refresh';

// API Services
import { GenerationAPI } from '../services/api';

function Generation({ checkpoints }) {
  const [selectedCheckpoint, setSelectedCheckpoint] = useState('');
  const [prompt, setPrompt] = useState('');
  const [negativePrompt, setNegativePrompt] = useState('');
  const [numImages, setNumImages] = useState(1);
  const [guidanceScale, setGuidanceScale] = useState(7.5);
  const [generatingImages, setGeneratingImages] = useState(false);
  const [error, setError] = useState('');
  const [generatedImages, setGeneratedImages] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedCheckpoint) {
      setError('Please select a checkpoint');
      return;
    }
    
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }
    
    // Reset messages
    setError('');
    setGeneratingImages(true);
    
    try {
      const response = await GenerationAPI.generateImages({
        dalle_checkpoint: selectedCheckpoint,
        prompt,
        negative_prompt: negativePrompt,
        num_images: numImages,
        temperature: guidanceScale
      });
      
      setGeneratedImages(response.images.map(image => ({
        url: image,
        prompt,
        negative_prompt: negativePrompt,
        timestamp: new Date().toISOString()
      })));
      
    } catch (error) {
      console.error('Generation error:', error);
      setError(error.response?.data?.error || 'Failed to generate images');
    } finally {
      setGeneratingImages(false);
    }
  };

  const handleDownload = (imageUrl) => {
    const link = document.createElement('a');
    link.href = imageUrl;
    link.download = `generated-${new Date().getTime()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleCopyPrompt = (text) => {
    navigator.clipboard.writeText(text);
  };

  // Example prompt suggestions for Notion-like experience
  const promptSuggestions = [
    "A photo of [subject] in Paris with the Eiffel Tower in the background",
    "A professional headshot of [subject] in formal attire",
    "A creative portrait of [subject] in the style of watercolor painting",
    "A cinematic shot of [subject] as a character in a sci-fi movie"
  ];

  return (
    <Paper
      elevation={0}
      sx={{
        p: 3,
        borderRadius: '8px',
        border: '1px solid',
        borderColor: 'divider',
      }}
    >
      <Typography variant="h5" sx={{ mb: 2, fontWeight: 500 }}>
        Generate Images
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
        Generate custom images using your trained models. Describe what you want to see in the prompt.
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={5}>
          <Box component="form" onSubmit={handleSubmit} noValidate>
            <FormControl fullWidth margin="normal" sx={{ mb: 3 }}>
              <InputLabel id="checkpoint-label">Select Model</InputLabel>
              <Select
                labelId="checkpoint-label"
                id="checkpoint"
                value={selectedCheckpoint}
                onChange={(e) => setSelectedCheckpoint(e.target.value)}
                label="Select Model"
                disabled={generatingImages || checkpoints.length === 0}
              >
                {checkpoints.length === 0 ? (
                  <MenuItem value="">No models available</MenuItem>
                ) : (
                  checkpoints.map((checkpoint) => (
                    <MenuItem key={checkpoint} value={checkpoint}>
                      {checkpoint}
                    </MenuItem>
                  ))
                )}
              </Select>
              <FormHelperText>
                Select a trained model to generate images with
              </FormHelperText>
            </FormControl>
            
            <TextField
              margin="normal"
              required
              fullWidth
              id="prompt"
              label="Prompt"
              name="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              multiline
              rows={3}
              placeholder="Describe what you want to see in the image"
              helperText="Use [subject] to refer to the trained person"
              sx={{ mb: 3 }}
            />
            
            {/* Prompt suggestions */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="caption" color="text.secondary" gutterBottom>
                Prompt suggestions:
              </Typography>
              <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap', gap: 1 }}>
                {promptSuggestions.map((suggestion, index) => (
                  <Tooltip title="Click to use this prompt" key={index}>
                    <Button
                      size="small"
                      variant="outlined"
                      onClick={() => setPrompt(suggestion)}
                      sx={{ 
                        textTransform: 'none', 
                        borderRadius: '16px',
                        px: 1.5,
                        py: 0.5,
                        minHeight: 0,
                        fontSize: '0.7rem'
                      }}
                    >
                      {suggestion.length > 30 ? suggestion.substring(0, 27) + '...' : suggestion}
                    </Button>
                  </Tooltip>
                ))}
              </Stack>
            </Box>
            
            <TextField
              margin="normal"
              fullWidth
              id="negative_prompt"
              label="Negative Prompt (Optional)"
              name="negative_prompt"
              value={negativePrompt}
              onChange={(e) => setNegativePrompt(e.target.value)}
              placeholder="Elements you want to avoid in the image"
              helperText="E.g., 'blurry, bad quality, distorted face'"
              sx={{ mb: 3 }}
            />
            
            <Typography variant="subtitle2" gutterBottom>
              Number of Images
            </Typography>
            <Slider
              value={numImages}
              onChange={(e, newValue) => setNumImages(newValue)}
              min={1}
              max={4}
              step={1}
              marks={[
                { value: 1, label: '1' },
                { value: 2, label: '2' },
                { value: 3, label: '3' },
                { value: 4, label: '4' }
              ]}
              valueLabelDisplay="auto"
              disabled={generatingImages}
              sx={{ mb: 3 }}
            />
            
            <Typography variant="subtitle2" gutterBottom>
              Guidance Scale
            </Typography>
            <Slider
              value={guidanceScale}
              onChange={(e, newValue) => setGuidanceScale(newValue)}
              min={1}
              max={15}
              step={0.5}
              marks={[
                { value: 1, label: 'Low' },
                { value: 7.5, label: 'Med' },
                { value: 15, label: 'High' }
              ]}
              valueLabelDisplay="auto"
              disabled={generatingImages}
              sx={{ mb: 4 }}
            />
            <FormHelperText sx={{ mt: -3, mb: 3 }}>
              How closely to follow the prompt (7-8 recommended)
            </FormHelperText>
            
            <Button
              type="submit"
              fullWidth
              variant="contained"
              disabled={generatingImages || !selectedCheckpoint || !prompt.trim()}
              startIcon={generatingImages ? <CircularProgress size={20} color="inherit" /> : <Image />}
              sx={{ py: 1.5 }}
            >
              {generatingImages ? 'Generating...' : 'Generate Images'}
            </Button>
          </Box>
        </Grid>

        <Grid item xs={12} md={7}>
          <Box sx={{ height: '100%' }}>
            {generatedImages.length > 0 ? (
              <Box>
                <Stack direction="row" justifyContent="space-between" alignItems="center" sx={{ mb: 2 }}>
                  <Typography variant="subtitle1">
                    Generated Images
                  </Typography>
                  <Button 
                    startIcon={<Refresh />} 
                    size="small" 
                    onClick={handleSubmit}
                    disabled={generatingImages}
                  >
                    Regenerate
                  </Button>
                </Stack>
                <Divider sx={{ mb: 2 }} />
                <Grid container spacing={2}>
                  {generatedImages.map((image, index) => (
                    <Grid item xs={12} sm={6} key={index}>
                      <Card 
                        elevation={0} 
                        sx={{ 
                          height: '100%', 
                          display: 'flex', 
                          flexDirection: 'column',
                          border: '1px solid',
                          borderColor: 'divider',
                          borderRadius: 2,
                          transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
                          '&:hover': {
                            transform: 'translateY(-4px)',
                            boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.1)'
                          }
                        }}
                      >
                        <CardMedia
                          component="img"
                          image={image.url}
                          alt={`Generated image ${index + 1}`}
                          sx={{ aspectRatio: '1/1', objectFit: 'cover' }}
                        />
                        <CardContent sx={{ flexGrow: 1, p: 1.5 }}>
                          <Typography variant="caption" color="text.secondary" sx={{ 
                            display: 'block', 
                            whiteSpace: 'nowrap',
                            overflow: 'hidden',
                            textOverflow: 'ellipsis'
                          }}>
                            {image.prompt}
                          </Typography>
                        </CardContent>
                        <CardActions sx={{ p: 1, pt: 0 }}>
                          <Tooltip title="Download image">
                            <IconButton onClick={() => handleDownload(image.url)} size="small">
                              <Download fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Copy prompt">
                            <IconButton onClick={() => handleCopyPrompt(image.prompt)} size="small">
                              <ContentCopy fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </CardActions>
                      </Card>
                    </Grid>
                  ))}
                </Grid>
              </Box>
            ) : (
              <Box 
                sx={{ 
                  height: '100%', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  flexDirection: 'column',
                  p: 4,
                  border: '1px dashed',
                  borderColor: 'divider',
                  borderRadius: 2
                }}
              >
                <Image sx={{ fontSize: 40, color: 'text.secondary', mb: 2 }} />
                <Typography variant="body2" color="text.secondary" align="center">
                  Your generated images will appear here
                </Typography>
                <Typography variant="caption" color="text.secondary" align="center" sx={{ mt: 1 }}>
                  Select a model, write a prompt, and click "Generate Images"
                </Typography>
              </Box>
            )}
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
}

export default Generation; 