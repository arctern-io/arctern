import React, {useEffect, useContext} from 'react';
import Button from '@material-ui/core/Button';
import {makeStyles} from '@material-ui/core/styles';
import {I18nContext} from '../../contexts/I18nContext';
import {CONFIG} from '../../utils/Consts';

const useStyles = makeStyles({
  title: {
    marginBottom: '10px !important',
    textTransform: 'uppercase',
  },
  content: {
    marginBottom: '20px',
    display: 'flex',
    justifyContent: 'first',
    alignItems: 'flex-end',
  },
  customRoot: {
    padding: '6px 0',
  },
});
const StackedShowType = (props: any) => {
  const {nls} = useContext(I18nContext);
  const classes = useStyles();
  const {config, setConfig} = props;

  const onClickv = () => {
    setConfig({type: CONFIG.ADD_STACKTYPE, payload: 'vertical'});
  };
  const onClickh = () => {
    setConfig({type: CONFIG.ADD_STACKTYPE, payload: 'horizontal'});
  };

  useEffect(() => {
    if (!config.stackType) {
      setConfig({type: CONFIG.ADD_STACKTYPE, payload: 'horizontal'});
    }
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <>
      <p className={classes.title}>
        {nls.label_widgetEditorDisplay_stackedDisplayType_displayType}
      </p>
      <div className={classes.content}>
        <Button
          size="medium"
          variant={config.stackType === 'horizontal' ? undefined : 'contained'}
          onClick={onClickv}
        >
          {nls.label_widgetEditorDisplay_stackedDisplayType_vertical}
        </Button>
        <Button
          size="medium"
          variant={config.stackType === 'vertical' ? undefined : 'contained'}
          onClick={onClickh}
        >
          {nls.label_widgetEditorDisplay_stackedDisplayType_horizontal}
        </Button>
      </div>
    </>
  );
};

export default StackedShowType;
