import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { ThemeProvider, createTheme } from '@mui/material/styles'
import CssBaseline from '@mui/material/CssBaseline'

// Import Roboto font
import '@fontsource/roboto/300.css'
import '@fontsource/roboto/400.css'
import '@fontsource/roboto/500.css'
import '@fontsource/roboto/700.css'

import App from './App.jsx'

// Create a Notion-inspired theme
const theme = createTheme({
  palette: {
    primary: {
      main: '#2383E2', // Notion blue
    },
    secondary: {
      main: '#EA4C89', // Notion pink for accents
    },
    background: {
      default: '#ffffff',
      paper: '#ffffff',
    },
    text: {
      primary: 'rgba(25, 25, 25, 0.95)',
      secondary: 'rgba(25, 25, 25, 0.6)',
    },
    divider: 'rgba(0, 0, 0, 0.06)',
  },
  typography: {
    fontFamily: "'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif",
    h1: {
      fontSize: '2.5rem',
      fontWeight: 500,
    },
    h2: {
      fontSize: '1.75rem',
      fontWeight: 500,
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 500,
    },
    body1: {
      fontSize: '1rem',
      lineHeight: 1.5,
    },
  },
  shape: {
    borderRadius: 3,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
  },
});

// Add dark mode support
const darkTheme = createTheme({
  ...theme,
  palette: {
    ...theme.palette,
    mode: 'dark',
    background: {
      default: '#191919',
      paper: '#262626',
    },
    text: {
      primary: 'rgba(255, 255, 255, 0.95)',
      secondary: 'rgba(255, 255, 255, 0.6)',
    },
    divider: 'rgba(255, 255, 255, 0.06)',
  },
});

// Use system preference for theme
const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
const activeTheme = prefersDarkMode ? darkTheme : theme;

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider theme={activeTheme}>
      <CssBaseline />
      <App />
    </ThemeProvider>
  </StrictMode>,
)
