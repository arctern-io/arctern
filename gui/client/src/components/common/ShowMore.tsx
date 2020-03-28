import React, {useContext, MouseEventHandler} from 'react';
import {makeStyles} from '@material-ui/core/styles';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import {customOptsStyle} from '../../utils/Theme';
import {I18nContext} from '../../contexts/I18nContext';

const useStyles = makeStyles({
  ...customOptsStyle,
});
interface IShowMoreProps {
  onClick: MouseEventHandler;
}
const ShowMore = (props: IShowMoreProps) => {
  const {onClick = () => {}} = props;
  const classes = useStyles();
  const {nls} = useContext(I18nContext);
  return (
    <ListItem classes={{gutters: classes.customGutters}} button divider>
      <ListItemText
        classes={{root: classes.customTextRoot}}
        primary={
          <span className={classes.option} onClick={onClick}>
            <span className={classes.optionLabel}>{nls.label_showMore}</span>
          </span>
        }
      />
    </ListItem>
  );
};

export default ShowMore;
