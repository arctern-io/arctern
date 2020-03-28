import React, {FC} from 'react';
import BarChart from '../BarChart';
import {BarChartProps} from './types';
import {CONFIG} from '../../utils/Consts';
import {barFilterHandler} from './filter';

const BarChartView: FC<BarChartProps> = props => {
  const {config, setConfig} = props;

  const onBarClick = (bar: any) => {
    setConfig({
      type: CONFIG.UPDATE,
      payload: barFilterHandler(config, bar),
    });
  };
  return <BarChart {...props} onBarClick={onBarClick} />;
};

export default BarChartView;
