import React, {FC, useState, useEffect, useRef} from 'react';
import {useTheme} from '@material-ui/core/styles';
import {isValidValue} from '../../utils/Helpers';
import {measureGetter} from '../../utils/WidgetHelpers';
import {getFitFontSize} from '../Utils/Decorators';
import {NO_DATA} from '../../utils/Consts';
import {formatterGetter} from '../../utils/Formatters';
import {NumberChartProps} from './types';
import './style.scss';

const NumberChart: FC<NumberChartProps> = props => {
  const {config, data, wrapperWidth, wrapperHeight} = props;
  const theme = useTheme();
  const container = useRef<any>(null);
  const {colorKey} = config;
  const valueMeasure = measureGetter(config, 'value')!;
  const hasData = data.length > 0;
  const [fontSize, setFontSize] = useState<any>(50);

  let dataV: any = NO_DATA;

  if (hasData && isValidValue(data[0][valueMeasure.as])) {
    const value = data[0][valueMeasure.as];
    const formatter = formatterGetter(valueMeasure);
    dataV = formatter(Number.parseFloat(value.toFixed(4)));
  }

  useEffect(() => {
    let width = wrapperWidth as number;
    let height = wrapperHeight as number;
    let newFontSize = getFitFontSize(dataV, width, height);
    setFontSize(newFontSize);
  }, [dataV, wrapperWidth, wrapperHeight]);

  return (
    <div
      className="z-chart z-number-chart"
      style={{
        color: dataV === NO_DATA ? '#666' : colorKey,
        fontSize: `${fontSize}px`,
        width: wrapperWidth,
        height: wrapperHeight,
        background: theme.palette.background.paper,
      }}
      ref={container}
    >
      {dataV}
    </div>
  );
};

export default NumberChart;
