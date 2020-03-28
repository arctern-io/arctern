import React, {FC} from 'react';
import {CONFIG} from '../../utils/Consts';
import PieChart from '../PieChart';
import {PieChartProps} from './types';
import {pieFilterHandler} from './filter';

const PieChartView: FC<PieChartProps> = props => {
  const {config, setConfig} = props;

  const onPieClick = (pie: any) => {
    setConfig({
      type: CONFIG.UPDATE,
      payload: pieFilterHandler(config, pie),
    });
  };
  return <PieChart {...props} onPieClick={onPieClick} />;
};

export default PieChartView;
