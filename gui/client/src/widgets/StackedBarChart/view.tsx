import React, {FC} from 'react';
import {StackedBarChartProps} from './types';
import {CONFIG} from '../../utils/Consts';
import StackedBarChart from '../StackedBarChart';
import {stackedBarFilterHander} from './filter';

const StackedBarChartView: FC<StackedBarChartProps> = props => {
  const {config, setConfig} = props;

  const onStackedBarClick = (data: any) => {
    setConfig({
      type: CONFIG.UPDATE,
      payload: stackedBarFilterHander(config, data),
    });
  };

  return (
    <>
      <StackedBarChart {...props} onStackedBarClick={onStackedBarClick} />
    </>
  );
};

export default StackedBarChartView;
