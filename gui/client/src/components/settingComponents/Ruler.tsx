import React, {useState, useEffect, useRef, useContext} from 'react';
import {I18nContext} from '../../contexts/I18nContext';
import {getValidRulerBase} from '../../utils/WidgetHelpers';
import Slider from '@material-ui/core/Slider';
import TextField from '@material-ui/core/TextField';
import {makeStyles} from '@material-ui/core/styles';
import {changeRangeSliderInputBox, changeRangeSlider} from '../../utils/EditorHelper';
import {DEFAULT_RULER} from '../../utils/Colors';
import {CONFIG} from '../../utils/Consts';

const useStyles = makeStyles({
  root: {
    display: 'flex',
    justifyContent: 'space-between',
  },
  inputColor: {
    display: 'flex',
    justifyContent: 'space-between',
  },
  customInput: {
    textAlign: 'center',
    width: '85px',
  },
});
const step = 0.01;
const RulerEditor = (props: any) => {
  const {nls} = useContext(I18nContext);
  const classes = useStyles({});
  const {config, setConfig, data = [], dataMeta} = props;
  const {ruler = DEFAULT_RULER, rulerBase = DEFAULT_RULER} = config;
  const [[minInput, maxInput], setMinMaxInput]: any = useState([
    Number.parseFloat(ruler.min),
    Number.parseFloat(ruler.max),
  ]);

  const delayCallback = ({validRange}: any) => {
    setMinMaxInput(validRange);
    setConfig({type: CONFIG.ADD_RULER, payload: {min: validRange[0], max: validRange[1]}});
  };
  const onMinInputChange = (e: any) => {
    const val = e.target.value * 1;
    const range = [rulerBase.min, ruler.max];
    const immediateCallback = (val: number) => setMinMaxInput([val, ruler.max]);
    changeRangeSliderInputBox({
      val,
      range,
      step,
      target: 'min',
      immediateCallback,
      delayCallback,
    });
  };

  const onMaxInputChange = (e: any) => {
    const val = e.target.value * 1;
    const range = [ruler.min, rulerBase.max];
    const immediateCallback = (val: number) => setMinMaxInput([ruler.min, val]);
    changeRangeSliderInputBox({
      val,
      range,
      step,
      target: 'max',
      immediateCallback,
      delayCallback,
    });
  };

  const onSlideChange = (e: any, val: any) => {
    const immediateCallback = (val: any[]) => {
      setMinMaxInput(val);
      config.ruler = {min: val[0], max: val[1]};
    };
    const delayCallback = (val: any) => {
      if (val[0] === val[1]) {
        val[1] = val[0] + 1;
      }
      setConfig({type: CONFIG.ADD_RULER, payload: {min: val[0], max: val[1]}});
    };
    changeRangeSlider({val, immediateCallback, delayCallback});
  };

  useEffect(() => {
    if (!config.ruler) {
      setConfig({type: CONFIG.ADD_RULER, payload: DEFAULT_RULER});
      setConfig({type: CONFIG.ADD_RULERBASE, payload: DEFAULT_RULER});
    }
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    const validRulerBase = getValidRulerBase({data, config});
    if (validRulerBase) {
      setConfig({type: CONFIG.ADD_RULER, payload: validRulerBase});
      setConfig({type: CONFIG.ADD_RULERBASE, payload: validRulerBase});
    }
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataMeta]);
  const isFirstRun = useRef<boolean>(true);
  useEffect(() => {
    if (isFirstRun.current) {
      isFirstRun.current = false;
      return;
    }
    setMinMaxInput([Number.parseFloat(rulerBase.min), Number.parseFloat(rulerBase.max)]);
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(rulerBase)]);

  return (
    <>
      <div className={classes.root}>
        <TextField
          classes={{root: classes.customInput}}
          label={nls.label_min_color}
          value={minInput}
          onChange={onMinInputChange}
          type="number"
          InputLabelProps={{
            shrink: true,
          }}
          margin="normal"
          inputProps={{style: {textAlign: 'center'}}}
        />
        <TextField
          classes={{root: classes.customInput}}
          label={nls.label_max_color}
          value={maxInput}
          onChange={onMaxInputChange}
          type="number"
          InputLabelProps={{
            shrink: true,
          }}
          margin="normal"
          inputProps={{style: {textAlign: 'center'}}}
        />
      </div>
      <Slider
        min={Number.parseFloat(rulerBase.min)}
        max={Number.parseFloat(rulerBase.max)}
        value={[minInput, maxInput]}
        onChange={onSlideChange}
        step={0.001}
      />
    </>
  );
};

export default RulerEditor;
