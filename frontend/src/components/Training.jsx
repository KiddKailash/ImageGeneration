import { useState, useEffect } from 'react';

// MUI components
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Button from '@mui/material/Button';
import Alert from '@mui/material/Alert';
import FormControl from '@mui/material/FormControl';
import InputLabel from '@mui/material/InputLabel';
import Select from '@mui/material/Select';
import MenuItem from '@mui/material/MenuItem';
import Slider from '@mui/material/Slider';
import FormHelperText from '@mui/material/FormHelperText';
import LinearProgress from '@mui/material/LinearProgress';
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';
import Divider from '@mui/material/Divider';
import Card from '@mui/material/Card';
import CardMedia from '@mui/material/CardMedia';
import CardContent from '@mui/material/CardContent';

// MUI icons
import Psychology from '@mui/icons-material/Psychology';
import SaveAlt from '@mui/icons-material/SaveAlt';
import Schedule from '@mui/icons-material/Schedule';

// API Services
import { SubjectAPI, TrainingAPI } from '../services/api';


function Training({ subjects, onSuccess }) {
  const [selectedSubject, setSelectedSubject] = useState('');
  const [trainingSteps, setTrainingSteps] = useState(150);
  const [learningRate, setLearningRate] = useState(0.0001);
  const [isTraining, setIsTraining] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [previewUrl, setPreviewUrl] = useState('');
  const [lastStep, setLastStep] = useState(0);
  const [subjectImages, setSubjectImages] = useState([]);

  useEffect(() => {
    if (selectedSubject) {
      fetchSubjectImages();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSubject]);

  const fetchSubjectImages = async () => {
    try {
      const response = await SubjectAPI.getSubjectImages(selectedSubject);
      setSubjectImages(response.images || []);
    } catch (error) {
      console.error('Error fetching images:', error);
      setSubjectImages([]);
    }
  };

  const handleSubjectChange = (e) => {
    setSelectedSubject(e.target.value);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!selectedSubject) {
      setError('Please select a subject');
      return;
    }
    
    // Reset messages
    setError('');
    setSuccess('');
    setProgress(0);
    setIsTraining(true);
    setLastStep(0);
    setPreviewUrl('');
    
    try {
      // Start training
      const trainingResponse = await TrainingAPI.startTraining({
        subject_token: selectedSubject,
        steps: trainingSteps,
        learning_rate: learningRate
      });
      
      const trainingId = trainingResponse.training_id;
      
      // Poll training status
      const pollInterval = setInterval(async () => {
        try {
          const statusResponse = await TrainingAPI.getTrainingStatus(trainingId);
          const { status, progress: currentProgress, preview_url, step } = statusResponse;
          
          setProgress(currentProgress);
          
          if (preview_url && step > lastStep) {
            setPreviewUrl(preview_url);
            setLastStep(step);
          }
          
          if (status === 'completed') {
            clearInterval(pollInterval);
            setSuccess(`Training completed successfully. Model saved as ${statusResponse.checkpoint_name}`);
            setIsTraining(false);
            
            // Call onSuccess callback with checkpoint name
            onSuccess(statusResponse.checkpoint_name);
          } else if (status === 'failed') {
            clearInterval(pollInterval);
            setError('Training failed. Please try again.');
            setIsTraining(false);
          }
        } catch (error) {
          console.error('Error polling training status:', error);
        }
      }, 2000);
      
    } catch (error) {
      console.error('Training error:', error);
      setError(error.response?.data?.error || 'Failed to start training');
      setIsTraining(false);
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
        Train Custom Model
      </Typography>
      
      <Typography variant="body2" color="text.secondary" sx={{ mb: 4 }}>
        Train a custom AI model to generate images featuring your subject in any style or setting.
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
      
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Box component="form" onSubmit={handleSubmit} noValidate>
            <FormControl fullWidth margin="normal" sx={{ mb: 3 }}>
              <InputLabel id="subject-label">Select Subject</InputLabel>
              <Select
                labelId="subject-label"
                id="subject"
                value={selectedSubject}
                onChange={handleSubjectChange}
                label="Select Subject"
                disabled={isTraining || subjects.length === 0}
              >
                {subjects.length === 0 ? (
                  <MenuItem value="">No subjects available</MenuItem>
                ) : (
                  subjects.map((subject) => (
                    <MenuItem key={subject} value={subject}>
                      {subject}
                    </MenuItem>
                  ))
                )}
              </Select>
              <FormHelperText>
                Select a subject that you've previously uploaded images for
              </FormHelperText>
            </FormControl>
            
            <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
              Training Steps
            </Typography>
            <Slider
              value={trainingSteps}
              onChange={(e, newValue) => setTrainingSteps(newValue)}
              min={50}
              max={300}
              step={10}
              marks={[
                { value: 50, label: '50' },
                { value: 150, label: '150' },
                { value: 300, label: '300' }
              ]}
              valueLabelDisplay="auto"
              disabled={isTraining}
              sx={{ mb: 3 }}
            />
            <FormHelperText sx={{ mt: -2, mb: 3 }}>
              More steps means better quality but longer training time (recommended: 150-200)
            </FormHelperText>
            
            <Typography variant="subtitle2" gutterBottom>
              Learning Rate
            </Typography>
            <Slider
              value={learningRate}
              onChange={(e, newValue) => setLearningRate(newValue)}
              min={0.00001}
              max={0.001}
              step={0.00001}
              marks={[
                { value: 0.00001, label: 'Low' },
                { value: 0.0001, label: 'Med' },
                { value: 0.001, label: 'High' }
              ]}
              valueLabelDisplay="auto"
              valueLabelFormat={(value) => value.toExponential(4)}
              disabled={isTraining}
              sx={{ mb: 3 }}
            />
            <FormHelperText sx={{ mt: -2, mb: 4 }}>
              Lower is more stable, higher can learn faster but might be less stable
            </FormHelperText>
            
            <Button
              type="submit"
              fullWidth
              variant="contained"
              disabled={isTraining || !selectedSubject}
              startIcon={<Psychology />}
              sx={{ py: 1.5 }}
            >
              {isTraining ? 'Training in Progress...' : 'Start Training'}
            </Button>
          </Box>
          
          {isTraining && (
            <Box sx={{ mt: 3 }}>
              <Stack direction="row" spacing={1} sx={{ mb: 1, alignItems: 'center' }}>
                <Schedule fontSize="small" color="primary" />
                <Typography variant="body2">
                  Training Progress: {Math.round(progress * 100)}%
                </Typography>
              </Stack>
              <LinearProgress variant="determinate" value={progress * 100} />
            </Box>
          )}
        </Grid>
        
        <Grid item xs={12} md={6}>
          <Box sx={{ 
            height: '100%', 
            display: 'flex', 
            flexDirection: 'column',
            justifyContent: 'flex-start'
          }}>
            {subjectImages.length > 0 ? (
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  Training Images
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <Grid container spacing={1}>
                  {subjectImages.slice(0, 8).map((image, index) => (
                    <Grid item xs={3} key={index}>
                      <Box
                        sx={{
                          width: '100%',
                          paddingTop: '100%',
                          position: 'relative',
                          borderRadius: 1,
                          overflow: 'hidden',
                          border: '1px solid',
                          borderColor: 'divider',
                        }}
                      >
                        <Box
                          component="img"
                          src={image}
                          alt={`Subject ${index + 1}`}
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
                {subjectImages.length > 8 && (
                  <Typography variant="caption" sx={{ display: 'block', mt: 1, textAlign: 'right' }}>
                    +{subjectImages.length - 8} more images
                  </Typography>
                )}
              </Box>
            ) : previewUrl ? (
              <Box>
                <Typography variant="subtitle1" gutterBottom>
                  Latest Preview
                </Typography>
                <Divider sx={{ mb: 2 }} />
                <Card elevation={0} sx={{ border: 1, borderColor: 'divider', borderRadius: 2 }}>
                  <CardMedia
                    component="img"
                    image={previewUrl}
                    alt="Training Preview"
                    sx={{ aspectRatio: '1/1' }}
                  />
                  <CardContent>
                    <Typography variant="body2" color="text.secondary">
                      Preview at step {lastStep}/{trainingSteps}
                    </Typography>
                  </CardContent>
                </Card>
              </Box>
            ) : (
              <Box 
                sx={{ 
                  height: '100%', 
                  display: 'flex', 
                  alignItems: 'center', 
                  justifyContent: 'center',
                  flexDirection: 'column',
                  p: 3,
                  border: '1px dashed',
                  borderColor: 'divider',
                  borderRadius: 2
                }}
              >
                <SaveAlt sx={{ fontSize: 40, color: 'text.secondary', mb: 2 }} />
                <Typography variant="body2" color="text.secondary" align="center">
                  Select a subject to see training images, or start training to see previews
                </Typography>
              </Box>
            )}
          </Box>
        </Grid>
      </Grid>
    </Paper>
  );
}

export default Training;