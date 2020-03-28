import React from 'react';
import DataInfoTable from './DataInfoTable';
import {useTheme, makeStyles} from '@material-ui/core/styles';
import WidgetView from './WidgetView';
import {genWidgetEditorStyle} from './index.style';
import './index.scss';

const useStyles = makeStyles(theme => genWidgetEditorStyle(theme) as any) as Function;

const WidgetEditorContent = (props: any) => {
  const theme = useTheme();
  const classes = useStyles(theme);
  const {status} = props;
  return (
    <>
      {status === 'showChart' ? (
        <WidgetView {...props} />
      ) : (
        <DataInfoTable {...props} title={status} />
      )}
      <div className={classes.bottom} />
    </>
  );
};

export default WidgetEditorContent;
