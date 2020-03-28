import React, {useContext} from 'react';
import {I18nContext} from '../../contexts/I18nContext';
import {makeStyles, useTheme} from '@material-ui/core/styles';
const useStyles = makeStyles(theme => ({
  noneSelect: {
    height: '30px',
    borderColor: theme.palette.grey[700],
    borderRadius: '5px',
    border: 'solid',
    color: theme.palette.text.disabled,
    borderWidth: '.5px',
    lineHeight: '30px',
    textAlign: 'center',
  },
}));

const NoSelector = () => {
  const theme = useTheme();
  const {nls} = useContext(I18nContext);
  const classes = useStyles(theme);
  return <div className={classes.noneSelect}>{nls.label_widgetEditor_noneRequired}</div>;
};

export default NoSelector;
