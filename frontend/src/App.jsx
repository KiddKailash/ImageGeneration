import { useState, useEffect } from 'react';
import axios from 'axios';

// Local Components
import Upload from './components/Upload';
import Training from './components/Training';
import Generation from './components/Generation';

// MUI Components
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Paper from '@mui/material/Paper';
import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';
import Chip from '@mui/material/Chip';
import Container from '@mui/material/Container';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import { useTheme } from '@mui/material/styles';
import useMediaQuery from '@mui/material/useMediaQuery';

// MUI Icons
import UploadIcon from '@mui/icons-material/CloudUpload';
import TrainingIcon from '@mui/icons-material/Psychology';
import GenerateIcon from '@mui/icons-material/Image';

// API Base URL - using environment variable with fallback
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

function App() {
  const [activeTab, setActiveTab] = useState('upload');
  const [serverStatus, setServerStatus] = useState('connecting');
  const [subjects, setSubjects] = useState([]);
  const [checkpoints, setCheckpoints] = useState([]);
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('sm'));

  // Check server status on mount
  useEffect(() => {
    const checkStatus = async () => {
      try {
        const response = await axios.get(`${API_URL}/status`);
        setServerStatus(response.data.status);
        
        // Fetch available subjects and checkpoints
        const subjectsResponse = await axios.get(`${API_URL}/subjects`);
        setSubjects(subjectsResponse.data.subjects || []);
        
        const checkpointsResponse = await axios.get(`${API_URL}/checkpoints`);
        setCheckpoints(checkpointsResponse.data.checkpoints || []);
      } catch (error) {
        console.error('Error connecting to server:', error);
        setServerStatus('offline');
      }
    };
    
    checkStatus();
  }, []);

  const handleChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
      {/* App Bar */}
      <AppBar position="static" color="transparent" elevation={0} sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Toolbar>
          <Typography variant="h5" color="inherit" sx={{ flexGrow: 1, fontWeight: 600 }}>
            DALL-E Image Generator
          </Typography>
          <Chip 
            label={`Server: ${serverStatus}`} 
            size="small"
            color={
              serverStatus === 'operational' ? 'success' : 
              serverStatus === 'connecting' ? 'warning' : 'error'
            }
            variant="outlined"
          />
        </Toolbar>
      </AppBar>

      {/* Main Content */}
      <Container maxWidth="lg" sx={{ flexGrow: 1, py: 4 }}>
        <Paper 
          elevation={0} 
          sx={{ 
            p: 3, 
            borderRadius: '8px', 
            border: '1px solid',
            borderColor: 'divider',
            mb: 4
          }}
        >
          <Typography variant="body1" color="text.secondary" paragraph>
            Train a custom AI model to generate images featuring you in any style or setting
          </Typography>
          
          {/* Navigation Tabs */}
          <Tabs 
            value={activeTab} 
            onChange={handleChange} 
            aria-label="app navigation"
            variant={isMobile ? "fullWidth" : "standard"}
            sx={{ mb: 2 }}
          >
            <Tab 
              icon={<UploadIcon />} 
              iconPosition="start" 
              label="Upload Images" 
              value="upload" 
            />
            <Tab 
              icon={<TrainingIcon />} 
              iconPosition="start" 
              label="Train Model" 
              value="train" 
            />
            <Tab 
              icon={<GenerateIcon />} 
              iconPosition="start" 
              label="Generate Images" 
              value="generate" 
            />
          </Tabs>
        </Paper>

        {/* Tab Panels */}
        <Box sx={{ mt: 2 }}>
          {activeTab === 'upload' && 
            <Upload 
              apiUrl={API_URL} 
              onSuccess={() => setActiveTab('train')} 
            />
          }
          {activeTab === 'train' && 
            <Training 
              apiUrl={API_URL} 
              subjects={subjects} 
              onSuccess={(newCheckpoint) => {
                setCheckpoints([...checkpoints, newCheckpoint]);
                setActiveTab('generate');
              }} 
            />
          }
          {activeTab === 'generate' && 
            <Generation 
              apiUrl={API_URL} 
              checkpoints={checkpoints} 
            />
          }
        </Box>
      </Container>

      {/* Footer */}
      <Box 
        component="footer" 
        sx={{ 
          py: 3, 
          px: 2, 
          mt: 'auto',
          borderTop: 1,
          borderColor: 'divider',
          textAlign: 'center'
        }}
      >
        <Typography variant="body2" color="text.secondary">
          Personal DALL-E Image Generator
        </Typography>
      </Box>
    </Box>
  );
}

export default App;
