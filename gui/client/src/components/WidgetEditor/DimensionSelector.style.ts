import {genBasicStyle} from '../../utils/Theme';

const genDimensionSelectorStyles = (theme: any) => {
  return {
    ...genBasicStyle(theme.palette.primary.main),
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
      fontSize: '12px',
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
    addDimension: {
      flexGrow: 1,
      alignItems: 'center',
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
    },
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
    option: {
      width: '100%',
      flexGrow: 1,
      display: 'flex',
      position: 'relative',
    },
    optionLabel: {
      flexGrow: 1,
      padding: '8px 16px',
    },
    icon: {
      zIndex: -10,
      position: 'absolute',
      width: '20px',
      top: '9px',
      right: '20px',
      textAlign: 'center',
      color: theme.palette.text.hint,
    },
    disable: {
      color: theme.palette.text.disable,
    },
    disableReq: {
      color: theme.palette.text.disable,
    },
    customGutters: {
      padding: '0px',
    },
    customTextRoot: {
      margin: 0,
      padding: 0,
    },
  };
};

export default genDimensionSelectorStyles;
