import React, {FC} from 'react';
import NumberChart from '../NumberChart';
import {DefaultWidgetProps} from '../../types';

const NumberChartView: FC<DefaultWidgetProps> = props => {
  return <NumberChart {...props} />;
};

export default NumberChartView;
