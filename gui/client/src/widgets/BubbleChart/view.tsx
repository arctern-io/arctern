import React from 'react';
import {cloneObj} from '../../utils/Helpers';
import {bubbleFilterHandler} from './filter';
import {BubbleChartProps} from './types';
import BubbleChart from '../BubbleChart';
import {CONFIG} from '../../utils/Consts';

const BubbleChartView = (props: BubbleChartProps) => {
  const {config, setConfig} = props;
  const cloneConfig = cloneObj(config);

  const onBubbleClick = (item: any) => {
    setConfig({
      type: CONFIG.UPDATE,
      payload: bubbleFilterHandler(cloneConfig, item),
    });
  };

  return <BubbleChart {...props} onClick={onBubbleClick} />;
};

export default BubbleChartView;
