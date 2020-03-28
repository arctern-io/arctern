import {genBasicStyle} from '../../utils/Theme';

export const genHeaderStyle = (theme: any) => {
  const baseColor = theme.palette.primary.main;
  const widgetBgColor = theme.palette.background.default;
  const borderColor = theme.palette.grey[900];
  return {
    ...genBasicStyle(baseColor),
    root: {
      padding: theme.spacing(2, 0, 1, 4),
      backgroundColor: widgetBgColor,
    },
    title: {
      display: 'flex',
      alignItems: 'center',
      margin: theme.spacing(0, 0, 2, 0),
    },
    editor: {
      display: 'flex',
    },
    tools: {
      display: 'flex',
      alignItems: 'center',
    },
    marginRight: {
      marginRight: theme.spacing(0.5),
    },
    tool: {
      position: 'relative',
      textAlign: 'center',
      marginRight: theme.spacing(0.5),
    },
    layout: {
      height: theme.spacing(5),
      width: theme.spacing(5),
      padding: theme.spacing(1, 1),
    },
    layoutTypes: {
      flexGrow: 1,
      display: 'flex',
      position: 'absolute',
      zIndex: 100,
      width: '200px',
      padding: theme.spacing(1),
      justifyContent: 'space-between',
      backgroundColor: widgetBgColor,
      borderColor: borderColor,
      border: 'solid',
      borderWidth: '.5px',
    },
    icon: {
      width: '25px',
    },
    filterNum: {
      position: 'absolute',
      top: '-6px',
      right: 0,
      zIndex: 100,
      width: '16px',
      height: '16px',
      backgroundColor: baseColor,
      color: theme.palette.text.primary,
      textAlign: 'center',
      borderRadius: '8px',
      lineHeight: '16px',
    },
    addBtn: {
      padding: '0 20px 0 0 ',
    },
    editHeader: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '20px',
      backgroundColor: widgetBgColor,
    },
    transform: {
      transform: 'rotate(180deg)',
    },
    hidden: {
      display: 'none',
    },
  };
};
