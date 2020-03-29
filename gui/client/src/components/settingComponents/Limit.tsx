import React, {useState, useEffect, ChangeEvent} from 'react';
import TextField from '@material-ui/core/TextField';
import Slider from '@material-ui/core/Slider';
import {makeStyles, useTheme} from '@material-ui/core/styles';
import {isValidValue} from '../../utils/Helpers';
import {changeInputBox, changeSlider} from '../../utils/EditorHelper';
import {WidgetConfig} from '../../types';
import {CONFIG} from '../../utils/Consts';

const useStyles = makeStyles(theme => ({
  title: {
    marginBottom: theme.spacing(2),
  },
  label: {
    marginBottom: theme.spacing(1),
  },
  color: {
    color: theme.palette.primary.main,
    width: '150px',
  },
  content: {
    marginBottom: theme.spacing(4),
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-end',
  },
  customInput: {
    textAlign: 'center',
    width: '85px',
  },
}));

interface ILimit {
  attr?: keyof WidgetConfig;
  title: string;
  initValue: number;
  config: WidgetConfig;
  setConfig: Function;
  min: number;
  max: number;
  step: number;
}
const getActionType = (attr: any) => {
  switch (attr) {
    case 'limit':
      return CONFIG.ADD_LIMIT;
    case 'points':
      return CONFIG.UPDATE_POINTS;
    case 'pointSize':
      return CONFIG.UPDATE_POINT_SIZE;
    default:
      return 0;
  }
};
const Limit = (props: ILimit) => {
  const theme = useTheme();
  const classes = useStyles(theme);
  const [inputNum, setInputNum]: any = useState();
  const {
    attr = 'limit',
    title,
    config,
    setConfig,
    min = 1000,
    max = 10000000,
    initValue = min * 2,
    step = 1,
  } = props;
  const value: number = config[attr] || min;
  const type = getActionType(attr);
  const delayCallback = (val: number) => {
    setConfig({type, payload: {[attr]: val}});
  };

  const onInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    changeInputBox({e, range: [min, max], immediateCallback: setInputNum, delayCallback});
  };

  const onSlideChange = (event: ChangeEvent<{}>, val: number | number[]) => {
    changeSlider({val, immediateCallback: setInputNum, delayCallback});
  };

  useEffect(() => {
    if (!config[attr]) {
      setConfig({payload: {limit: initValue}, type});
    }
    setInputNum();
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value]);

  return (
    <>
      <p className={classes.title}>{title}</p>
      <div className={classes.content}>
        <Slider
          classes={{root: classes.color}}
          min={min}
          max={max}
          value={inputNum || value}
          onChange={onSlideChange}
          step={step}
        />
        <TextField
          classes={{root: classes.customInput}}
          value={isValidValue(inputNum) ? inputNum : value}
          type="number"
          onChange={onInputChange}
          InputLabelProps={{
            shrink: true,
          }}
          margin="normal"
          // eslint-disable-next-line
          inputProps={{style: {textAlign: 'center'}}}
        />
      </div>
    </>
  );
};

export default Limit;
