import {genBasicStyle, customOptsStyle} from '../../utils/Theme';

const genMeasureSelectorStyles = (theme: any) => ({
  ...genBasicStyle(theme.palette.primary.main),
  ...customOptsStyle,
  root: {
    height: '31px',
    lineHeight: '30px',
    marginBottom: '10px',
    border: 'solid',
    display: 'flex',
    alignItems: 'center',
    textAlign: 'center',
    fontSize: '12px',
    borderColor: theme.palette.grey[700],
    borderWidth: '.5px',
    borderRadius: '5px',
    position: 'relative',
  },
  button: {
    margin: 0,
    padding: 0,
    flexGrow: 0,
  },
  short: {
    width: '50px',
    borderTopLeftRadius: '5px',
    borderBottomLeftRadius: '5px',
    backgroundColor: theme.palette.grey[400],
    fontSize: '12px',
    textTransform: 'uppercase',
    textAlign: 'center',
  },
  content: {
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'start',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
    flexGrow: 1,
    paddingLeft: '10px',
  },
  deleteSeletor: {
    width: '20px',
    height: '29px',
  },
  addMeasure: {
    flexGrow: 1,
    alignItems: 'center',
    overflow: 'hidden',
    textOverflow: 'ellipsis',
    whiteSpace: 'nowrap',
  },
  clearInput: {},
  hidden: {
    display: 'none',
  },
  input: {
    marginBottom: '10px',
  },
  options: {
    maxHeight: '150px',
    overflowY: 'auto',
    margin: 0,
    padding: 0,
    listStyle: 'none',
  },
  disable: {
    color: theme.palette.text.disable,
  },
  disableReq: {
    color: theme.palette.text.disable,
  },
});

export default genMeasureSelectorStyles;
