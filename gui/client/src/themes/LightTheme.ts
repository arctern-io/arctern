import BLUE from '@material-ui/core/colors/blue';
import RED from '@material-ui/core/colors/red';
import GREY from '@material-ui/core/colors/grey';
import {addOverRide} from './common';

const LightTheme: any = {
  palette: {
    type: 'light',
    primary: {
      main: BLUE[700],
      contrastText: GREY[900],
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
      800: GREY[500],
      900: GREY[900],
      A200: GREY[500],
    },
    text: {
      primary: '#000',
      disabled: GREY[600],
      hint: GREY[600],
    },
    background: {
      default: '#fff',
      paper: GREY[200],
    },
  },
};

export default addOverRide(LightTheme);
