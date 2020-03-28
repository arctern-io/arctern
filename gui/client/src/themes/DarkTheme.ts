import BLUE from '@material-ui/core/colors/blue';
import RED from '@material-ui/core/colors/red';
import GREY from '@material-ui/core/colors/grey';
import {addOverRide} from './common';

const DarkTheme: any = {
  palette: {
    type: 'dark', // Switching the dark mode on is a single property value change.
    primary: {
      main: BLUE[500],
      contrastText: GREY[50],
    },
    secondary: {
      main: RED.A200,
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
      primary: GREY[50],
      disabled: GREY[600],
      hint: GREY[600],
    },
    background: {
      default: '#252525',
      paper: GREY[900],
    },
  },
};

export default addOverRide(DarkTheme);
