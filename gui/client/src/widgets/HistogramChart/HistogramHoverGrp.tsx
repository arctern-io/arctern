import React, {FC, useRef} from 'react';
import {OUT_OUT_CHART} from '../../utils/Consts';

const HoverGrp: FC<any> = props => {
  const {height, hoverData, barWidth} = props;
  const ref: any = useRef();
  const CENTER = barWidth > 1 ? barWidth / 2 : 1;
  if (hoverData && hoverData.x !== OUT_OUT_CHART && hoverData.data.length) {
    return (
      <g
        ref={ref}
        transform={`translate(${hoverData.x + CENTER},0)`}
        className="hoverGrp"
        fill="none"
        pointerEvents="all"
        stroke="#FFFFFF"
      >
        <line className="vertical-marker" y1="0" y2={height} />
      </g>
    );
  }

  return <></>;
};

export default HoverGrp;
