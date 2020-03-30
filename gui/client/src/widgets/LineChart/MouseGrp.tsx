import React, {FC, useState, useEffect, useRef, useContext} from 'react';
import {bisector, select, event, mouse} from 'd3';
import HoverGrp from './HoverGrp';
import Brush from '../common/Brush';
import TimeLinePlayer from './TimelinePlayer';
import {rootContext} from '../../contexts/RootContext';
import {defaultChartMousePos} from '../../utils/Consts';
import {
  seriesHoverDataGetter,
  dimensionTypeGetter,
  typeDistanceGetter,
  typeEqGetter,
  dimensionGetter,
} from '../../utils/WidgetHelpers';

const MouseGrp: FC<any> = props => {
  const {showTooltip, hideTooltip} = useContext(rootContext);
  const {
    isRange,
    config,
    svgWidth,
    x,
    y,
    svgHeight,
    margin,
    width,
    height,
    tooltipTitleGetter,
    tooltipContentGetter,
    legendItems,
    seriesData,
    xDomain,
    onRangeChange,
    setTimeSelection,
  } = props;
  const ref: any = useRef();
  // mouse position state
  const [position, setPosition] = useState<any>(defaultChartMousePos);
  // get mouse position
  const effectFactors = [width, height, xDomain];
  useEffect(() => {
    let _refNode = select(ref.current);
    _refNode.on('mousemove', () => {
      let [xPos, yPos] = mouse(event.target);
      const outOfBound = xPos < 0 || yPos < 0;
      setPosition(
        outOfBound
          ? defaultChartMousePos
          : {
              x: xPos > width ? width : xPos,
              y: yPos > height ? height : yPos,
              xV: x.invert(xPos),
              yV: y.invert(yPos),
              event: event,
            }
      );
    });

    _refNode.on('mouseout', () => {
      setPosition(defaultChartMousePos);
    });

    return () => {
      _refNode.on('mousemove', null);
      _refNode.on('mouseout', null);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(effectFactors)]);

  // get hover data
  const typeX = dimensionTypeGetter(dimensionGetter(config, 'x')!);
  const eq = typeEqGetter(typeX);
  const distance = typeDistanceGetter(typeX);
  const hoverData = seriesHoverDataGetter(
    legendItems,
    seriesData,
    position,
    x,
    y,
    bisector((d: any) => d.x).left,
    distance,
    eq
  );
  if (position.x === defaultChartMousePos.x) {
    hideTooltip();
  } else {
    showTooltip({
      position,
      tooltipData: hoverData,
      titleGetter: tooltipTitleGetter,
      contentGetter: tooltipContentGetter,
    });
  }

  return (
    <>
      <svg
        width={svgWidth}
        height={svgHeight}
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          pointerEvents: 'none',
        }}
      >
        <g
          ref={ref}
          width={width}
          height={height}
          transform={`translate(${margin.left}, ${margin.top})`}
        >
          <rect fill="none" width={width} height={height} style={{pointerEvents: 'visible'}} />
          <HoverGrp height={height} hoverData={hoverData} />
          <Brush
            {...props}
            filter={isRange ? config.selfFilter : config.filter}
            onBrushUpdate={(range: any) => {
              onRangeChange(config, range, isRange);
              setTimeSelection(range);
            }}
            onBrush={setTimeSelection}
          />
        </g>
      </svg>
      {!isRange && (
        <TimeLinePlayer
          {...props}
          x={x}
          onPlay={(range: any) => {
            onRangeChange(config, range);
          }}
        />
      )}
    </>
  );
};

export default MouseGrp;
