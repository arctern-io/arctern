export const genWidgetEditorStyle = (theme: any) => {
  return {
    root: {
      flexGrow: 1,
      position: 'relative',
      width: '100%',
      minWidth: '1000px',
      height: 'calc(100% - 0px)',
      backgroundColor: theme.palette.background.paper,
    },
    sidebar: {
      position: 'absolute',
      width: '300px',
      height: 'calc(100% - 0px)',
      overflowY: 'auto',
      padding: '10px 20px 10px 10px',
      textAlign: 'left',
      zIndex: 100,
    },
    sidebarLeft: {
      top: 0,
      left: 0,
    },
    sidebarRight: {
      top: 0,
      right: 0,
    },
    chart: {
      position: 'absolute',
      width: '100%',
      marginTop: '20px',
      padding: '0 300px',
      height: 'calc(100% - 20px)',
      overflowY: 'hidden',
      display: 'flex',
      flexDirection: 'column',
      zIndex: 10,
    },
    input: {
      textAlign: 'center',
    },
    sources: {
      padding: '5px',
      marginBottom: '40px',
      '& p': {padding: 0, margin: 0},
    },
    widgetTitle: {
      fontSize: '1rem',
      padding: '3px 0 5px',
      textAlign: 'center',
      fontWeight: 'bold',
      cursor: 'pointer',
      position: 'relative',
      backgroundColor: theme.palette.background.default,
    },
    title: {
      marginBottom: '10px',
      textTransform: 'uppercase',
    },
    widgetContent: {
      flexGrow: 1,
      display: 'flex',
      flexDirection: 'column',
      overflowY: 'auto',
      backgroundColor: theme.palette.background.default,
    },
    bottom: {
      width: '100%',
      height: '40px',
      backgroundColor: theme.palette.background.paper,
    },
    bigIcon: {
      width: '400px',
      margin: 'auto',
      flexGrow: 2,
      position: 'relative',
    },
    requirement: {
      textAlign: 'center',
      flexGrow: 1,
      color: theme.palette.primary.main,
      flexDirection: 'column',
      marginBottom: '10px',
      backgroundPosition: '50% 10%',
      backgroundSize: `30%`,
      backgroundRepeat: 'no-repeat',
    },
    loading: {
      position: 'absolute',
      left: 0,
      width: '100%',
      height: '100%',
      zIndex: 1000,
      display: 'flex',
      alignItems: 'center',
      backgroundColor: 'transparent',
      pointerEvents: 'all',
      marginBottom: '10px',
    },
    label: {
      color: 'white',
      marginTop: '40%',
    },
    colNamePreview: {
      flexGrow: 1,
      display: 'flex',
      justifyContent: 'space-between',
      backgroundColor: theme.palette.background.default,
    },
    baseInfo: {
      flex: 1,
      padding: '10px',
    },
    baseInfoTitle: {
      color: theme.palette.grey[900],
      fontWeight: 'bold',
    },
    baseInfoWrapper: {
      marginBottom: '20px',
    },
    baseInfoDetail: {
      margin: 0,
      padding: 0,
      wordWrap: 'break-word',
    },
    colNames: {
      flex: 1,
      overflowY: 'auto',
    },
    showTitle: {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '32px',
    },
    customDeleteIcon: {
      position: 'absolute',
      left: '8px',
      width: '16px',
      top: '4px',
      cursor: 'pointer',
    },
  };
};

export const genCustomSettingStyle = (theme: any) => {
  return {
    root: {
      padding: '5px',
      marginBottom: '40px',
      '& p': {padding: 0, margin: 0},
    },
    source: {
      marginBottom: '40px',
    },
    chartStyle: {
      width: '120px',
      height: '30px',
    },
    title: {
      marginBottom: '10px',
      textTransform: 'uppercase',
    },
    formatItem: {
      marginBottom: '20px',
    },
    colorRange: {
      display: 'flex',
      justifyContent: 'start',
    },
    colorItem: {
      width: '26px',
      height: '26px',
      display: 'inline-block',
      padding: '1px',
      borderColor: theme.palette.grey[700],
      borderWidth: '.5px',
      marginRight: '5px',
      cursor: 'pointer',
    },
    selected: {
      border: 'solid',
    },
  };
};
