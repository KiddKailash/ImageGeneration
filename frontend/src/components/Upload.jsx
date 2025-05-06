import { useState } from 'react';
import axios from 'axios';

// MUI components
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import TextField from '@mui/material/TextField';
import Button from '@mui/material/Button';
import Alert from '@mui/material/Alert';
import Grid from '@mui/material/Grid';
import { styled } from '@mui/material/styles';

// MUI icons
import PhotoCamera from '@mui/icons-material/PhotoCamera';
import UploadIcon from '@mui/icons-material/Upload';


// Styled components for file input
const VisuallyHiddenInput = styled('input')({
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  height: 1,
  overflow: 'hidden',
  position: 'absolute',
  bottom: 0,
  left: 0,
  whiteSpace: 'nowrap',
  width: 1,
});

function Upload({ apiUrl, onSuccess }) {
  const [subjectToken, setSubjectToken] = useState('');
  const [files, setFiles] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [previewUrls, setPreviewUrls] = useState([]);

  const handleFileChange = (e) => {
    const selectedFiles = Array.from(e.target.files);
    setFiles(selectedFiles);
    
    // Generate preview URLs
    const newPreviewUrls = selectedFiles.map(file => URL.createObjectURL(file));
    setPreviewUrls(newPreviewUrls);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!subjectToken.trim()) {
      setError('Please enter a subject token');
      return;
    }
    
    if (files.length === 0) {
      setError('Please select at least one image');
      return;
    }
    
    // Reset messages
    setError('');
    setSuccess('');
    setUploading(true);
    
    // Create form data
    const formData = new FormData();
    formData.append('subject_token', subjectToken);
    files.forEach(file => formData.append('images', file));
    
    try {
      const response = await axios.post(`${apiUrl}/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      
      setSuccess(`Successfully uploaded ${response.data.image_count} images for ${response.data.subject_token}`);
      
      // Clear form
      setFiles([]);
      setPreviewUrls([]);
      
      // Call onSuccess callback after a delay
      setTimeout(() => {
        onSuccess();
      }, 2000);
      
    } catch (error) {
      console.error('Upload error:', error);
      setError(error.response?.data?.error || 'Failed to upload images');
    } finally {
      setUploading(false);
    }
  };

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
        Upload Training Images
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
        Upload 5-20 photos of a person to train the AI. Photos should be well-lit, clear,
        and show different angles or expressions.
      </Typography>
      
      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}
      
      {success && (
        <Alert severity="success" sx={{ mb: 3 }}>
          {success}
        </Alert>
      )}
      
      <Box component="form" onSubmit={handleSubmit} noValidate>
        <TextField
          margin="normal"
          required
          fullWidth
          id="subjectToken"
          label="Subject Token (e.g., 'john' or 'sarah')"
          name="subjectToken"
          value={subjectToken}
          onChange={(e) => setSubjectToken(e.target.value)}
          helperText="This token will be used in prompts when generating images (e.g., 'a photo of john in paris')"
          sx={{ mb: 3 }}
        />
        
        <Button
          component="label"
          variant="outlined"
          startIcon={<PhotoCamera />}
          sx={{ 
            mb: 3,
            p: 1.5,
            borderStyle: 'dashed',
            height: '100px',
            width: '100%',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center'
          }}
        >
          <Typography variant="body1" sx={{ mb: 1 }}>
            Click to select images
          </Typography>
          <Typography variant="caption" color="text.secondary">
            Accept .jpg, .jpeg, .png
          </Typography>
          <VisuallyHiddenInput 
            type="file" 
            multiple 
            accept="image/jpeg,image/png"
            onChange={handleFileChange}
          />
        </Button>
        
        {previewUrls.length > 0 && (
          <Box sx={{ mb: 3 }}>
            <Typography variant="subtitle2" sx={{ mb: 1 }}>
              Image Previews ({previewUrls.length})
            </Typography>
            <Grid container spacing={1}>
              {previewUrls.map((url, index) => (
                <Grid item xs={6} sm={4} md={3} key={index}>
                  <Box
                    sx={{
                      position: 'relative',
                      paddingTop: '100%', // 1:1 aspect ratio
                      overflow: 'hidden',
                      borderRadius: 1,
                      border: '1px solid',
                      borderColor: 'divider',
                    }}
                  >
                    <Box
                      component="img"
                      src={url}
                      alt={`Preview ${index + 1}`}
                      sx={{
                        position: 'absolute',
                        top: 0,
                        left: 0,
                        width: '100%',
                        height: '100%',
                        objectFit: 'cover',
                      }}
                    />
                  </Box>
                </Grid>
              ))}
            </Grid>
          </Box>
        )}
        
        <Button
          type="submit"
          fullWidth
          variant="contained"
          disabled={uploading}
          startIcon={<UploadIcon />}
          sx={{ 
            py: 1.5,
            backgroundColor: uploading ? undefined : theme => theme.palette.primary.main
          }}
        >
          {uploading ? 'Uploading...' : 'Upload Images'}
        </Button>
      </Box>
    </Paper>
  );
}

export default Upload; 