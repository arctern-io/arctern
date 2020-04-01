export const addOverRide = (theme: any) => {
  const BaseColor = theme.palette.primary.main;
  return {
    ...theme,
    overrides: {
      MuiTabs: {
        indicator: {
          color: BaseColor,
          backgroundColor: BaseColor,
        },
      },
      MuiTab: {
        root: {
          minWidth: '100px !important',
        },
        textColorInherit: {
          '&$selected': {
            color: BaseColor,
          },
        },
      },
      MuiIconButton: {
        root: {
          '&$disabled': {color: theme.palette.grey[700]},
        },
      },
      MuiButton: {
        root: {
          minWidth: 0,
        },
        contained: {
          backgroundColor: '#fff',
          color: theme.palette.text.primary,
        },
      },
    },
  };
};
