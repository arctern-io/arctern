import React, {FC, useState, useEffect, useRef, useContext} from 'react';
import {useTheme, makeStyles} from '@material-ui/core/styles';
import {queryContext} from '../../contexts/QueryContext';
import {I18nContext} from '../../contexts/I18nContext';
import {BinProps} from '../../types';
import Spinner from '../common/Spinner';
import Select from '@material-ui/core/Select';
import TextField from '@material-ui/core/TextField';
import MenuItem from '@material-ui/core/MenuItem';
import Slider from '@material-ui/core/Slider';
import Tabs from '@material-ui/core/Tabs';
import Tab from '@material-ui/core/Tab';
import {timeBinMap} from '../../utils/Time';
import {isDateCol} from '../../utils/ColTypes';
import {
  genRangeQuery,
  changeInputBox,
  changeSlider,
  changeRangeSliderInputBox,
  changeRangeSlider,
} from '../../utils/EditorHelper';
import {getValidTimeBinOpts} from '../../utils/Binning';
import {cloneObj, isValidValue} from '../../utils/Helpers';
import {sideBarItemWitdh} from '../../utils/Theme';
import BinStyles from './Bin.style';

const NumMinMaxStep = 0.01;
const MaxbinsRange = [2, 250];

const useStyles = makeStyles(BinStyles as any);
const Bin: FC<BinProps> = props => {
  const {getData} = useContext(queryContext);
  const {nls} = useContext(I18nContext);
  const theme = useTheme();
  const classes = useStyles(theme);
  const {dimension, addDimension, source, onAdd, staticRange = []} = props;
  const {
    min: currMin = 0,
    max: currMax = 100,
    extract = false,
    timeBin = 'day',
    maxbins = 12,
    type = 'DATE',
    extent = [1, 100],
  } = dimension;
  const [staticMin, staticMax] = staticRange;
  const binType = isDateCol(type) ? 'dateBin' : 'numBin';
  const dateBin = extract ? 'extract' : 'binning';
  const cloneDimension = cloneObj(dimension);
  const [isLoading, setIsLoading] = useState(false);
  const [range, setRange]: any = useState([currMin, currMax]);
  const [numMinInput, setNumMinInput]: any = useState();
  const [numMaxInput, setNumMaxInput]: any = useState();
  const [maxbinsInput, setMaxbinsInput]: any = useState();
  const [minimum, maximum] = range;

  const unitOpts: any = getValidTimeBinOpts(staticMin, staticMax, extract);
  const currUnitOpt = unitOpts.find((item: any) => item.value === timeBin) || {
    value: '',
  };

  const binSize = `${nls.label_widgetEditor_binSize} ${(
    (typeof currMax === 'number' && typeof currMin === 'number' ? currMax - currMin : 100) / maxbins
  ).toFixed(2)}`;
  // change to dateBin || dateBin attribute change. NumBin change won't use this effect
  const isFirstRun = useRef<boolean>(true);
  useEffect(() => {
    if (isFirstRun.current) {
      isFirstRun.current = false;
      return;
    }
    const isDateType = isDateCol(dimension.type);
    if (!isDateType) {
      return;
    }
    const params = genRangeQuery(dimension.value, source);
    getData(params).then((res: any) => {
      const {minimum = 0, maximum = 200} = res[0] || {};
      addDimension(
        {
          ...cloneDimension,
          min: cloneDimension.currMin,
          max: cloneDimension.currMax,
        },
        onAdd
      );

      setIsLoading(false);
      setRange([minimum, maximum]);
    });

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dimension.timeBin, dimension.extract, isDateCol(dimension.type)]);

  // change date type dimension
  const onTabChange = (e: any, val: any) => {
    if (dateBin === val) {
      return;
    }
    const _validOpts = getValidTimeBinOpts(staticMin, staticMax, val !== 'binning');
    const autoOpt = _validOpts[Math.floor(_validOpts.length / 2)];
    const _autoOpt = cloneObj(autoOpt);
    [_autoOpt.min, _autoOpt.max] = [_autoOpt.min || 1, _autoOpt.max || 30];
    addDimension(
      val === 'binning'
        ? {
            ...cloneDimension,
            binningResolution: timeBinMap[_autoOpt.value],
            extract: false,
            format: 'auto',
            timeBin: _autoOpt.value,
            min: dimension.min,
            max: dimension.max,
          }
        : {
            ...cloneDimension,
            extract: true,
            format: 'auto',
            timeBin: _autoOpt.value,
            min: _autoOpt.min,
            max: _autoOpt.max,
          }
    );
  };

  const onBinOptChange = (e: any, child: any) => {
    let val = child.props.value;
    const newDimension = {
      ...cloneDimension,
      binningResolution: timeBinMap[val],
      extract: false,
      format: 'auto',
      timeBin: val,
      min: minimum,
      max: maximum,
    };
    addDimension(newDimension);
  };

  const onExtractOptChange = (e: any, child: any) => {
    let val = child.props.value;
    let currMin: any = dimension.min,
      currMax: any = dimension.max;
    switch (val) {
      case 'year':
        currMin = new Date().getFullYear();
        currMax = new Date().getFullYear();
        break;

      case 'quarter':
        currMin = 1;
        currMax = 4;
        break;

      case 'month':
        currMin = 1;
        currMax = 12;
        break;

      case 'day':
        currMin = 1;
        currMax = 31;
        break;

      case 'isodow':
        currMin = 1;
        currMax = 7;
        break;

      case 'hour':
        val = 'hour';
        currMin = 0;
        currMax = 23;
        break;
      case 'minute':
      default:
        currMin = 1;
        currMax = 59;
        break;
    }

    const newDimension = {
      ...cloneDimension,
      binningResolution: timeBinMap[val],
      extract: true,
      timeBin: val,
      min: currMin,
      max: currMax,
      currMin,
      currMax,
    };
    addDimension(newDimension);
  };

  // change num type dimension
  // change extent
  const _rangeDelayCallback = ({validRange, target}: any) => {
    target === 'min' ? setNumMinInput(validRange[0]) : setNumMaxInput(validRange[1]);
    const newDimension = {
      ...cloneDimension,
      extent: validRange,
    };
    addDimension(newDimension);
  };
  const onNumMinChange = (e: any) => {
    const val = e.target.value * 1;
    const range = [minimum, extent[1]];
    const immediateCallback = (val: number) => setNumMinInput(val);
    changeRangeSliderInputBox({
      val,
      range,
      step: NumMinMaxStep,
      target: 'min',
      immediateCallback,
      delayCallback: _rangeDelayCallback,
    });
  };

  const onNumMaxChange = (e: any) => {
    const val = e.target.value * 1;
    const range = [extent[0], maximum];
    const immediateCallback = (val: number) => setNumMaxInput(val);
    changeRangeSliderInputBox({
      val,
      range,
      step: NumMinMaxStep,
      target: 'max',
      immediateCallback,
      delayCallback: _rangeDelayCallback,
    });
  };

  const onNumSliderChange = (e: any, val: any) => {
    const immediateCallback = (val: any[]) => {
      setNumMaxInput(val[1]);
      setNumMinInput(val[0]);
    };
    const delayCallback = (val: any[]) => {
      const newDimension = {
        ...cloneDimension,
        extent: [val[0], val[1]],
      };
      addDimension(newDimension);
    };
    changeRangeSlider({val, immediateCallback, delayCallback});
  };

  // change maxbins
  const _maxbinsDelayCallback = (val: number) => {
    const newDimension = {
      ...cloneDimension,
      maxbins: val,
    };
    addDimension(newDimension);
  };

  const onMaxbinsSliderChange = (e: any, val: any) => {
    changeSlider({
      val,
      immediateCallback: setMaxbinsInput,
      delayCallback: _maxbinsDelayCallback,
    });
  };

  const onMaxbinsInputChange = (e: any) => {
    changeInputBox({
      e,
      range: MaxbinsRange,
      immediateCallback: setMaxbinsInput,
      delayCallback: _maxbinsDelayCallback,
    });
  };

  return (
    <div className={classes.binRoot}>
      {isLoading && (
        <div style={{textAlign: 'center'}}>
          <Spinner />
        </div>
      )}
      {!isLoading && (
        <>
          {binType === 'dateBin' && (
            <>
              <Tabs
                className={classes.tabsLayout}
                value={dateBin}
                variant="fullWidth"
                onChange={onTabChange}
              >
                <Tab label={nls.label_widgetEditor_binning} value="binning" />
                <Tab label={nls.label_widgetEditor_Extract} value="extract" />
              </Tabs>
              <div>
                <div className={classes.binTitle}>
                  {`${
                    dateBin === 'binning'
                      ? nls.label_widgetEditor_bin
                      : nls.label_widgetEditor_extract
                  } ${nls.label_widgetEditor_unit}`}
                </div>
                <Select
                  style={{width: sideBarItemWitdh}}
                  value={currUnitOpt.value}
                  onChange={dateBin === 'binning' ? onBinOptChange : onExtractOptChange}
                >
                  {unitOpts.map((option: any, index: number) => (
                    <MenuItem className="bin-toot" key={index} value={option.value}>
                      {nls[`label_widgetEditor_binOpt_${option.label}`]}
                    </MenuItem>
                  ))}
                </Select>
              </div>
            </>
          )}
          {binType === 'numBin' && (
            <>
              <div className={classes.content}>
                <span>{binSize}</span>
                <TextField
                  classes={{root: classes.customInput}}
                  value={isValidValue(maxbinsInput) ? maxbinsInput : maxbins}
                  onChange={onMaxbinsInputChange}
                  type="number"
                  inputProps={{style: {textAlign: 'center'}}}
                />
              </div>
              <Slider
                className={classes.content}
                min={MaxbinsRange[0]}
                max={MaxbinsRange[1]}
                value={maxbinsInput || maxbins}
                onChange={onMaxbinsSliderChange}
              />
              <div className={classes.content}>
                <TextField
                  classes={{root: classes.customInput}}
                  value={numMinInput || extent[0]}
                  type="number"
                  onChange={onNumMinChange}
                  inputProps={{style: {textAlign: 'center'}}}
                />
                <span>{nls.label_widgetEditor_minMax}</span>
                <TextField
                  classes={{root: classes.customInput}}
                  value={numMaxInput || extent[1]}
                  type="number"
                  onChange={onNumMaxChange}
                  inputProps={{style: {textAlign: 'center'}}}
                />
              </div>
              <Slider
                classes={{root: classes.color}}
                min={Number.parseFloat(minimum)}
                max={Number.parseFloat(maximum)}
                value={[numMinInput || extent[0], numMaxInput || extent[1]]}
                onChange={onNumSliderChange}
                step={NumMinMaxStep}
              />
            </>
          )}
        </>
      )}
    </div>
  );
};

export default Bin;
