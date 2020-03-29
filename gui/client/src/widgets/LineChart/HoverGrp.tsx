import React, {FC, useRef} from 'react';
import {OUT_OUT_CHART} from '../../utils/Consts';
import {color} from '../../utils/Colors';

const HoverGrp: FC<any> = props => {
  const {height, hoverData} = props;
  const ref: any = useRef();
  const markerR: any = 3;
  // console.log(hoverData)
  if (hoverData && hoverData.x !== OUT_OUT_CHART && hoverData.data.length) {
    const circles = hoverData.data.filter((d: any) => d);
    return (
      <g
        ref={ref}
        transform={`translate(${hoverData.x},0)`}
        className="hoverGrp"
        fill="none"
        pointerEvents="all"
        stroke="#FFFFFF"
      >
        <line className="vertical-marker" y1="0" y2={height} />
        {circles.map((d: any, i: number) => {
          return (
            <circle
              key={i}
              className="dot"
              cy={d.yPos}
              r={markerR}
              stroke={d.color || color(d.as)}
              fill={d.color || color(d.as)}
            />
          );
        })}
      </g>
    );
  }

  return <></>;
};

export default HoverGrp;
