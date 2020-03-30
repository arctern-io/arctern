import React, {useRef, useState, useEffect, useContext, useMemo} from 'react';
import Brush from '../common/Brush';
import HoverGrp from './HistogramHoverGrp';
import {defaultChartMousePos, COLUMN_TYPE} from '../../utils/Consts';
import {select, event, mouse} from 'd3';
import {
  histogramHoverGetter,
  dimensionTypeGetter,
  typeDistanceGetter,
  typeEqGetter,
  dimensionGetter,
} from '../../utils/WidgetHelpers';
import {rootContext} from '../../contexts/RootContext';

const HistogramChartMouseGrp = (props: any) => {
  const {showTooltip, hideTooltip} = useContext(rootContext);
  const {
    config,
    svgWidth,
    svgHeight,
    margin,
    x,
    y,
    xDomain,
    onRangeChange,
    renderData,
    barWidth,
    legendItems,
    isRangeChart,
    dataMeta,
    linkMeta,
    tooltipTitleGetter,
    tooltipContentGetter,
  } = props;
  const width = svgWidth - margin.left - margin.right,
    height = svgHeight - margin.top - margin.bottom;
  const ref: any = useRef();
  // mouse position state
  const [position, setPosition] = useState<any>(defaultChartMousePos);
  // get mouse position
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
  }, [width, height, JSON.stringify(xDomain)]);

  const xDimension = dimensionGetter(config, 'x')!;
  // get hover data
  const typeX = dimensionTypeGetter(xDimension);
  const isTimeChart: boolean = typeX === COLUMN_TYPE.DATE;

  const hoverData = useMemo(() => {
    const eq = typeEqGetter(typeX);
    const distance = typeDistanceGetter(typeX);
    const hoverData = histogramHoverGetter(legendItems, {y: renderData}, position, y, distance, eq);
    return hoverData;
  }, [typeX, legendItems, renderData, position, y]);

  useEffect(() => {
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
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [position, hoverData]);

  const findNearestData = (range: any) => {
    return range.map((r: number) => {
      let diff = r;
      return renderData.reduce((pre: number, cur: any) => {
        let positionX = cur[xDimension.as];
        if (isTimeChart) {
          positionX = new Date(positionX).getTime();
          r = new Date(r).getTime();
        }
        const newDiff = Math.abs(positionX - r);
        pre = newDiff < diff ? positionX : pre;
        diff = newDiff < diff ? newDiff : diff;
        return pre;
      }, r);
    });
  };

  const handleBrushUpdate = (range: any) => {
    onRangeChange(config, findNearestData(range), !!isRangeChart);
  };

  return (
    <svg
      width={svgWidth}
      height={svgHeight}
      style={{
        position: 'absolute',
        left: 0,
        top: 0,
      }}
    >
      <g
        width={width}
        height={height}
        transform={`translate(${margin.left}, ${margin.top})`}
        ref={ref}
      >
        {/* toolBar */}

        <rect fill="none" width={width} height={height} style={{pointerEvents: 'visible'}} />
        <HoverGrp height={height} hoverData={hoverData} barWidth={barWidth} />
        {/* brush */}

        <Brush
          x={x}
          dataMeta={dataMeta}
          linkMeta={linkMeta}
          config={config}
          isTimeChart={isTimeChart}
          xDimension={xDimension}
          width={width}
          height={height}
          filter={isRangeChart ? config.selfFilter : config.filter}
          onBrushUpdate={handleBrushUpdate}
        />
      </g>
    </svg>
  );
};

export default HistogramChartMouseGrp;
