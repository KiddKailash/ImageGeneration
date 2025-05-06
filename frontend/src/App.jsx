import { useState, useEffect } from 'react';

// Local Components
import Upload from './components/Upload';
import Training from './components/Training';
import Generation from './components/Generation';

// API Services
import { ServerAPI, SubjectAPI, TrainingAPI } from './services/api';

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
        const statusResponse = await ServerAPI.getStatus();
        setServerStatus(statusResponse.status);
        
        // Fetch available subjects and checkpoints
        const subjectsResponse = await SubjectAPI.getSubjects();
        setSubjects(subjectsResponse.subjects || []);
        
        const checkpointsResponse = await TrainingAPI.getCheckpoints();
        setCheckpoints(checkpointsResponse.checkpoints || []);
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
              serverStatus === 'online' ? 'success' : 
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
              onSuccess={() => setActiveTab('train')} 
            />
          }
          {activeTab === 'train' && 
            <Training 
              subjects={subjects} 
              onSuccess={(newCheckpoint) => {
                setCheckpoints([...checkpoints, newCheckpoint]);
                setActiveTab('generate');
              }} 
            />
          }
          {activeTab === 'generate' && 
            <Generation 
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
