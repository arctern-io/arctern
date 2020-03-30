// import RED from '@material-ui/core/colors/red';
import GREY from '@material-ui/core/colors/grey';
import {addOverRide} from './common';

const Main = '#4FC4F9';
const CustomTheme: any = {
  palette: {
    type: 'light', // Switching the dark mode on is a single property value change.
    primary: {
      main: Main,
      contrastText: GREY[50],
    },
    secondary: {
      main: GREY[900],
    },
    grey: {
      50: GREY[50],
      100: GREY[100],
      200: GREY[200],
      300: GREY[300],
      400: GREY[400],
      500: GREY[500],
      600: GREY[600],
      700: GREY[700],
      800: GREY[800],
      900: GREY[900],
      A200: GREY[500],
    },
    text: {
      primary: GREY[900],
      disabled: GREY[600],
      hint: GREY[600],
    },
    background: {
      default: 'rgba(245,245,245,1)',
      paper: `rgba(229,229,229,1)`,
    },
    action: {
      disabled: 'red',
    },
  },
};

export default addOverRide(CustomTheme);
