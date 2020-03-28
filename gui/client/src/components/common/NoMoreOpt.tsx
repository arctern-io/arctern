import React, {useContext, MouseEventHandler} from 'react';
import {createStyles, makeStyles, useTheme} from '@material-ui/core/styles';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import {I18nContext} from '../../contexts/I18nContext';

const useStyles = makeStyles(theme =>
  createStyles({
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
    optionLabel: {
      flexGrow: 1,
      padding: '8px 24px 8px 16px',
      color: theme.palette.text.hint,
      fontStyle: 'italic',
      cursor: 'not-allowed',
    },
    customIcon: {
      zIndex: -10,
      position: 'absolute',
      right: '20px',
      top: '8px',
      textAlign: 'center',
    },
  })
);

interface INoMoreOpt {
  onClick?: MouseEventHandler;
}

const NoMoreOpt = (props: INoMoreOpt) => {
  const {onClick = () => {}} = props;
  const theme = useTheme();
  const classes = useStyles(theme);
  const {nls} = useContext(I18nContext);
  return (
    <ListItem classes={{gutters: classes.customGutters}} button divider>
      <ListItemText
        classes={{root: classes.customTextRoot}}
        primary={
          <span className={classes.option} onClick={onClick}>
            <span className={classes.optionLabel}>{nls.label_noother_opt}</span>
          </span>
        }
      />
    </ListItem>
  );
};

export default NoMoreOpt;
