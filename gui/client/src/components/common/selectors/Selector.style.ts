import {genBasicStyle, customOptsStyle} from '../../../utils/Theme';

const genSelectorStyle = (theme: any) => {
  const baseColor = theme.palette.primary.main;
  const tipColor = theme.palette.text.hint;
  const borderColor = theme.palette.grey[700];
  return {
    ...genBasicStyle(baseColor),
    ...customOptsStyle,
    root: {
      height: '31px',
      lineHeight: '30px',
      marginBottom: '10px',
      border: 'solid',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      textAlign: 'center',
      fontSize: '14px',
      borderColor: borderColor,
      borderWidth: '.5px',
      borderRadius: '5px',
    },
    buttonStatus: {
      flexGrow: 1,
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      textAlign: 'center',
    },
    content: {
      flexGrow: 1,
      position: 'relative',
      display: 'flex',
      alignItems: 'center',
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
      textAlign: 'center',
      justifyContent: 'start',
      paddingLeft: '15px',
    },
    onlyContent: {
      flexGrow: 1,
      display: 'flex',
      alignItems: 'center',
      overflow: 'hidden',
      textOverflow: 'ellipsis',
      whiteSpace: 'nowrap',
      textAlign: 'center',
      justifyContent: 'center',
    },
    bin: {
      paddingLeft: '4px',
    },
    value: {
      paddingLeft: '4px',
      textAlign: 'left',
    },
    deleteSeletor: {
      // backgroundColor: deleteColor,
      height: '29px',
      marginRight: '.5px',
      borderTopRightRadius: '5px',
      borderBottomRightRadius: '5px',
    },
    clearInput: {},
    hidden: {
      display: 'none',
    },
    input: {
      height: '30px',
      width: '100%',
      borderRadius: '5px',
    },
    options: {
      maxHeight: '150px',
      overflowY: 'auto',
      margin: 0,
      padding: 0,
      listStyle: 'none',
      alignItems: 'center',
    },
    contentSpan: {
      flexGrow: 1,
      padding: '8px 16px',
    },
    tip: {
      position: 'absolute',
      right: '10px',
      top: '50%',
      transform: 'translate(-50%,-50%)',
      color: tipColor,
      zIndex: -100,
      fontSize: '12px',
      boldWeight: '900',
    },
    clear: {
      display: 'flex',
      flexGrow: 1,
      justifyContent: 'center',
      alignItems: 'center',
      maxWidth: '40px',
    },
    customListItemRoot: {
      padding: '2px 0px',
      display: 'flex',
      flexGrow: 1,
    },
    icon: {
      zIndex: -10,
      position: 'absolute',
      right: '0px',
      width: '20px',
      top: '9px',
      textAlign: 'center',
    },
  };
};

export default genSelectorStyle;
