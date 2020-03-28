import React from 'react';
import {makeStyles, useTheme} from '@material-ui/core/styles';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import {genBasicStyle, customOptsStyle} from '../../utils/Theme';

const useStyles = makeStyles(
  theme =>
    ({
      ...genBasicStyle(theme.palette.primary.main),
      ...customOptsStyle,
      customTextRoot: {
        margin: 0,
        padding: 0,
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
    } as any)
) as Function;

const RecordSqlOpt = (props: any) => {
  const theme = useTheme();
  const classes = useStyles(theme);
  const {placeholder, onClick} = props;
  return (
    <ListItem classes={{gutters: classes.customGutters}} button divider>
      <ListItemText
        classes={{root: classes.customTextRoot}}
        primary={
          <span className={classes.option} onMouseDown={onClick}>
            <span className={classes.optionLabel}>{placeholder}</span>
            <span className={classes.customIcon}>#</span>
          </span>
        }
      />
    </ListItem>
  );
};

export default RecordSqlOpt;
