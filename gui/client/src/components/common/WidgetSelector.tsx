import React, {FC, Suspense, useContext} from 'react';
import clsx from 'clsx';
import {WidgetSelectorProps} from '../../types/Editor';
import Spinner from './Spinner';
import {makeStyles, useTheme} from '@material-ui/core/styles';
import {genBasicStyle} from '../../utils/Theme';
import {I18nContext} from '../../contexts/I18nContext';

const useStyles = makeStyles(theme => ({
  ...genBasicStyle(theme.palette.primary.main),
  editHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingBottom: '20px',
  },
  widgetType: {
    minWidth: '48px',
    height: '52px',
    borderWidth: '.5px',
    borderColor: theme.palette.grey[700],
    borderRadius: '3px',
    padding: '2px 6px',
    border: 'solid',
    display: 'flex',
    flexDirection: 'column',
    cursor: 'pointer',
  },
  element: {
    textAlign: 'center',
    margin: 0,
    padding: 0,
    fontSize: '12px',
    '& svg': {
      fill: 'currentColor',
      width: '1em',
      height: '1em',
      display: 'inline-block',
      fontSize: '1.5rem',
      transition: 'fill 200ms cubic-bezier(0.4, 0, 0.2, 1) 0ms',
    },
  },
  selected: {
    color: theme.palette.primary.main,
    borderColor: theme.palette.primary.main,
  },
}));

const WidgetSelector: FC<WidgetSelectorProps> = props => {
  const {icon, widgetType, selected, onClick} = props;
  const theme = useTheme();
  const classes = useStyles(theme);
  const {nls} = useContext(I18nContext);
  const onSelectorClick = () => {
    onClick(widgetType);
  };
  const label = nls[`label_Header_${widgetType}`] || widgetType;
  return (
    <Suspense fallback={<Spinner />}>
      <div
        className={clsx(classes.widgetType, classes.hover, selected ? classes.selected : '')}
        onClick={onSelectorClick}
      >
        <div className={classes.element} dangerouslySetInnerHTML={{__html: icon}} />
        <p className={classes.element}>{label}</p>
      </div>
    </Suspense>
  );
};

export default WidgetSelector;
