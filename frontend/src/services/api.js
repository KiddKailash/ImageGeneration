import axios from 'axios';

// Create an axios instance with default config
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:5001',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  }
});

// Subject related API calls
export const SubjectAPI = {
  // Get all subjects
  getSubjects: async () => {
    const response = await apiClient.get('/subjects');
    return response.data;
  },
  
  // Get training images for a specific subject
  getSubjectImages: async (subjectName) => {
    const response = await apiClient.get(`/training-images/${subjectName}`);
    return response.data;
  },
  
  // Upload images for a subject
  uploadImages: async (subjectName, formData) => {
    const response = await apiClient.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }
};

// Training related API calls
export const TrainingAPI = {
  // Start training for a subject
  startTraining: async (trainingData) => {
    const response = await apiClient.post('/train', trainingData);
    return response.data;
  },
  
  // Get training status
  getTrainingStatus: async (trainingId) => {
    const response = await apiClient.get(`/training_status/${trainingId}`);
    return response.data;
  },
  
  // Get all available checkpoints
  getCheckpoints: async () => {
    const response = await apiClient.get('/checkpoints');
    return response.data;
  }
};

// Image generation API calls
export const GenerationAPI = {
  // Generate images
  generateImages: async (generationData) => {
    const response = await apiClient.post('/generate', generationData);
    return response.data;
  },
  
  // Get generated image by filename
  getGeneratedImage: (filename) => {
    return `${apiClient.defaults.baseURL}/images/${filename}`;
  }
};

// Server status API call
export const ServerAPI = {
  getStatus: async () => {
    const response = await apiClient.get('/status');
    return response.data;
  }
};

export default {
  SubjectAPI,
  TrainingAPI,
  GenerationAPI,
  ServerAPI
}; 