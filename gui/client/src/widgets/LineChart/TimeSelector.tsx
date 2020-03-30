import React, {FC} from 'react';
import {dateFormat} from '../../utils/Formatters';

interface ITimeSelectorProps {
  timeSelectorData: any;
  timeSelection: any[];
  timeFormatter?: Function;
}

// time formatter
const TimeSelector: FC<ITimeSelectorProps> = props => {
  const {timeSelectorData, timeSelection, timeFormatter = dateFormat} = props;

  if (timeSelectorData.showTimeSelector) {
    const [xMin, xMax] = timeSelectorData.domain;
    let a: Date = timeSelection[0] || timeSelectorData.selectTime[0] || xMin;
    let b: Date = timeSelection[1] || timeSelectorData.selectTime[1] || xMax;

    return (
      <div className="timeSelector">
        {timeFormatter(new Date(a), '%H:%M %p %m/%d/%Y')}-
        {timeFormatter(new Date(b), '%H:%M %p %m/%d/%Y')}
      </div>
    );
  }

  return <></>;
};

export default TimeSelector;
