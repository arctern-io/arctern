import React, {useContext} from 'react';
import Button from '@material-ui/core/Button';
import {makeStyles} from '@material-ui/core/styles';
import {I18nContext} from '../../contexts/I18nContext';
import {CONFIG} from '../../utils/Consts';

const useStyles = makeStyles({
  title: {
    marginBottom: '10px !important',
    textTransform: 'uppercase',
  },
});

const StyleSwitch = (props: any) => {
  const {nls} = useContext(I18nContext);
  const classes = useStyles();
  const {config, setConfig} = props;

  const handleChange = () => {
    setConfig({type: CONFIG.CHANGE_IS_AREA, payload: !!!config.isArea});
  };

  return (
    <div>
      <p className={classes.title}>{nls.label_chart_style}</p>
      <Button
        size="medium"
        variant={config.isArea ? undefined : 'contained'}
        onClick={handleChange}
      >
        {nls.label_chart_style_line}
      </Button>
      <Button
        size="medium"
        variant={!config.isArea ? undefined : 'contained'}
        onClick={handleChange}
      >
        {nls.label_chart_style_area}
      </Button>
    </div>
  );
};

export default StyleSwitch;
