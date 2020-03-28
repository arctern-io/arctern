export const sideBarItemWitdh = '270px';
export const titleMarginBottom = '10px';
export const subTitleMarginBottom = '5px';
export const contentMarginBottom = '3px';
export const colorItemHeight = '30px';
export const colorItemSelectedPadding = '3px';

export const genBasicStyle = (baseColor: string) => {
  return {
    hover: {
      cursor: 'pointer',
      '&:hover': {
        color: baseColor,
        borderColor: baseColor,
      },
    },
  };
};

export const customOptsStyle: any = {
  customGutters: {
    padding: '0px',
  },
  customTextRoot: {
    margin: 0,
    padding: 0,
  },
  option: {
    width: '100%',
    flexGrow: 1,
    display: 'flex',
    position: 'relative',
  },
  optionLabel: {flexGrow: 1, padding: '8px 24px 8px 16px'},
  customIcon: {
    zIndex: -10,
    position: 'absolute',
    right: '20px',
    top: '8px',
    textAlign: 'center',
    // color: tipColor,
    fontSize: '12px',
  },
};
