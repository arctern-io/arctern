import React, {FC, useState, useEffect, useRef, useContext} from 'react';
import {useTheme} from '@material-ui/core/styles';
import {scaleBand, select} from 'd3';
import {rootContext} from '../../contexts/RootContext';
import GradientRuler from '../common/GradientRuler';
import {dimensionGetter, measureGetter, dimensionDataGetter} from '../../utils/WidgetHelpers';
import {typeSortGetter} from '../../utils/Sorts';
import {rangeFormatter, formatterGetter} from '../../utils/Formatters';
import {dimensionsDataToFilterExpr} from '../../utils/Filters';
import {UNSELECTED_COLOR, genColorGetter} from '../../utils/Colors';
import {DEFAULT_WIDGET_WRAPPER_WIDTH, DEFAULT_WIDGET_WRAPPER_HEIGHT} from '../../utils/Layout';
import {HeatChartProps} from './type';
import './style.scss';

// CHART realted consts
const margin = {top: 10, right: 20, bottom: 0, left: 40};
const X_LEGEND_HEIGHT = 35;
const X_LEGEND_MARGIN_TOP = 10;
const XTICK_MIN_HEIGHT = 40;
const XTICK_MAX_HEIGHT = 80;
const YTICK_MIN_WIDTH = 40;
const YTICK_MAX_WIDTH = 80;
const RECT_MIN_SIZE = 15;
const TICK_FONT_SIZE = 10 * 1; // size * line-height

const HeatChart: FC<HeatChartProps> = props => {
  const theme = useTheme();
  const {showTooltip, hideTooltip} = useContext(rootContext);
  const {
    config,
    data,
    dataMeta,
    onCellClick,
    onRowClick,
    onColClick,
    wrapperWidth = DEFAULT_WIDGET_WRAPPER_WIDTH,
    wrapperHeight = DEFAULT_WIDGET_WRAPPER_HEIGHT,
  } = props;

  const [view, setView] = useState<any>({
    width: 0,
    height: 0,
    adjustedSvgWidth: 0,
    adjustedSvgHeight: 0,
    rectWidth: 0,
    rectHeight: 0,
    yTickWidth: 0,
    xTickHeight: 0,
    xTicks: [],
    yTicks: [],
    x: scaleBand(),
    y: scaleBand(),
    dimentionData: [],
    xDimensionsData: [],
    yDimensionsData: [],
  });

  const getColor = genColorGetter(config);
  // get x dimension, get xType, get xAxis scale
  const xDimension = dimensionGetter(config, 'x')!;
  const yDimension = dimensionGetter(config, 'y')!;
  const colorMeasure = measureGetter(config, 'color')!;

  // x , y sorter
  const xSorter = typeSortGetter(xDimension.type, 'x');
  const ySorter = typeSortGetter(yDimension.type, 'y');

  // formatter
  const colorMeasureFormatter = (v: any) => {
    return `${colorMeasure.value}: ${formatterGetter(colorMeasure)(v)}`;
  };

  const effectors = JSON.stringify([
    dataMeta,
    wrapperWidth,
    wrapperHeight,
    config.dimensions,
    config.measures,
    config.filter,
  ]);

  useEffect(() => {
    const loading = dataMeta && dataMeta.loading;
    if (loading || wrapperHeight === 0 || wrapperWidth === 0) {
      return;
    }
    // get scale
    const x = scaleBand();
    const y = scaleBand();

    //  get bin data
    const dimentionData = dimensionDataGetter(config.dimensions, data).map(d => {
      let currentFilterExpr = dimensionsDataToFilterExpr(d.dimensionsData);
      let filters = config.filter;
      let filterNames = Object.keys(filters);
      d.data = {};
      d.data.filters = [];
      d.data.selected =
        filterNames.length === 0
          ? true
          : filterNames.filter((f: any) => {
              if (filters[f].expr === currentFilterExpr) {
                d.data.filters.push(f);
                return true;
              } else {
                return false;
              }
            }).length > 0;
      d.data.dimensionsData = d.dimensionsData;
      d.data.x = rangeFormatter([d.dimensionsData[0]]);
      d.data.y = rangeFormatter([d.dimensionsData[1]]);
      d.data.value = d[colorMeasure.as];
      return d;
    });
    const xData = dimentionData.sort(xSorter);
    const yData = dimentionData.sort(ySorter);

    // domain
    x.domain(data.sort(xSorter).map((d: any) => d.x));
    y.domain(data.sort(ySorter).map((d: any) => d.y));

    // ticks
    let xTickHeight = XTICK_MIN_HEIGHT;
    const xDimensionsData: any[] = [];
    const xTicks = x.domain().map((xValue: any) => {
      const obj = xData.filter((xDataObj: any) => xDataObj.x === xValue)[0];
      const _xDimensionsData = [obj.dimensionsData[0]];
      const tick = rangeFormatter(_xDimensionsData);

      xTickHeight = Math.max(tick.length * TICK_FONT_SIZE, xTickHeight);
      xDimensionsData.push(_xDimensionsData);
      return {x: obj.x, tick, dimensionsData: _xDimensionsData};
    });

    // calculate xtick height
    xTickHeight =
      xTickHeight > XTICK_MAX_HEIGHT
        ? XTICK_MAX_HEIGHT
        : xTickHeight < XTICK_MIN_HEIGHT
        ? XTICK_MIN_HEIGHT
        : xTickHeight;

    let yTickWidth = YTICK_MIN_WIDTH;
    const yDimensionsData: any[] = [];
    const yTicks = y.domain().map((yValue: any) => {
      const obj: any = yData.filter((yDataObj: any) => yDataObj.y === yValue)[0];
      const _yDimensionsData = [obj.dimensionsData[1]];
      const tick = rangeFormatter(_yDimensionsData);

      yTickWidth = Math.max(tick.length * TICK_FONT_SIZE, yTickWidth);
      yDimensionsData.push(_yDimensionsData);
      return {y: obj.y, tick, dimensionsData: _yDimensionsData};
    });

    // calculate yTick Width
    yTickWidth =
      yTickWidth > YTICK_MAX_WIDTH
        ? YTICK_MAX_WIDTH
        : yTickWidth < YTICK_MIN_WIDTH
        ? YTICK_MIN_WIDTH
        : yTickWidth;

    // chart size
    const xTickLen = x.domain().length;
    const yTickLen = y.domain().length;
    const svgWidth = wrapperWidth;
    const svgHeight = wrapperHeight - xTickHeight - X_LEGEND_HEIGHT - X_LEGEND_MARGIN_TOP;
    const width = svgWidth - margin.left - margin.right - yTickWidth,
      height = svgHeight - margin.top - margin.bottom;

    // rect size
    const rectWidth: number = Math.max(width / xTickLen, RECT_MIN_SIZE);
    const rectHeight: number = Math.max(height / yTickLen, RECT_MIN_SIZE);
    // recalculate width/height
    const adjustedWidth = rectWidth === Infinity ? width : rectWidth * xTickLen;
    const adjustedHeight = rectHeight === Infinity ? height : rectHeight * yTickLen;
    const adjustedSvgWidth = adjustedWidth + margin.left + margin.right + yTickWidth;
    const adjustedSvgHeight = adjustedHeight + margin.top + margin.bottom;
    // set scale range
    x.range([0, adjustedWidth]);
    y.range([adjustedHeight, 0]);

    const newView = {
      width,
      height,
      adjustedSvgWidth,
      adjustedSvgHeight,
      rectWidth,
      rectHeight,
      yTickWidth,
      xTickHeight,
      xTicks,
      yTicks: yTicks.reverse(), // from top to bottom
      x,
      y,
      dimentionData,
      xDimensionsData,
      yDimensionsData,
    };

    // update view
    setView(newView);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectors]);

  // animation
  const rectContainer = useRef<SVGGElement>(null);
  useEffect(() => {
    if (rectContainer.current) {
      // use d3 to do the transition, react handle the dom part
      const rects: any[] = Array.from(rectContainer.current!.children || []);

      rects.forEach((v: any) => {
        const dataObj = view.dimentionData[v.dataset.index];
        const width = view.rectWidth;
        const height = view.rectHeight;
        const x = view.x(dataObj.x);
        const y = view.y(dataObj.y);
        const fillColor: string = dataObj.data.selected
          ? getColor(dataObj[colorMeasure.as]) || theme.palette.background.default
          : UNSELECTED_COLOR;
        select(v)
          .transition()
          .duration(200)
          .attr('width', width)
          .attr('height', height)
          .attr('x', x)
          .attr('y', y)
          .attr('fill', fillColor)
          .attr('stroke', fillColor);
      });
    }

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify([view, config.colorKey, config.ruler])]);

  // bind events
  const _onCellClick = (dataObject: any) => {
    onCellClick && onCellClick(dataObject);
  };

  const _onRowClick = (yDimensionData: any) => {
    const merge = view.xDimensionsData.map((xDimensionData: any) => [
      ...xDimensionData,
      ...yDimensionData,
    ]);
    onRowClick && onRowClick(merge);
  };

  const _onColClick = (xDimenionData: any) => {
    const merge = view.yDimensionsData.map((yDimensionData: any) => [
      ...xDimenionData,
      ...yDimensionData,
    ]);
    onColClick && onColClick(merge);
  };

  // tooltip
  const tooltipTitleGetter = (hoverDatas: any) => {
    const title = `${hoverDatas.x} x ${hoverDatas.y}`;
    return <>{title}</>;
  };

  const tooltipContentGetter = (hoverDatas: any) => {
    return (
      <ul
        style={{
          listStyleType: 'none',
          padding: 0,
        }}
      >
        <li>
          <span
            className="mark"
            style={{
              background: getColor(hoverDatas.value) as string,
            }}
          />
          {colorMeasureFormatter(hoverDatas.value)}
        </li>
      </ul>
    );
  };

  const _onMouseMove = (e: any) => {
    const newTooltipData = {
      position: {
        event: e,
      },
      tooltipData: {
        x: e.target.dataset.yValue,
        y: e.target.dataset.xValue,
        value: e.target.dataset.measureValue,
      },
      titleGetter: tooltipTitleGetter,
      contentGetter: tooltipContentGetter,
    };

    showTooltip(newTooltipData);
  };

  const _onMouseLeave = () => {
    hideTooltip();
  };

  // render
  return (
    <div
      className={`z-chart z-heat-chart`}
      style={{
        width: wrapperWidth,
        height: wrapperHeight,
        background: theme.palette.background.paper,
      }}
    >
      <div className="container">
        <svg width={view.adjustedSvgWidth} height={view.adjustedSvgHeight}>
          <g
            width={view.adjustedWidth}
            height={view.adjustedHeight}
            transform={`translate(${margin.left + view.yTickWidth}, ${margin.top})`}
            onMouseLeave={_onMouseLeave}
            ref={rectContainer}
          >
            {view.dimentionData.map((dataObject: any, index: number) => {
              return (
                <rect
                  fill={UNSELECTED_COLOR}
                  stroke={UNSELECTED_COLOR}
                  key={index}
                  onClick={() => {
                    _onCellClick(dataObject);
                  }}
                  onMouseMove={_onMouseMove}
                  data-index={index}
                  data-x-value={dataObject.data.x}
                  data-y-value={dataObject.data.y}
                  data-measure-value={dataObject.data.value}
                  cursor="pointer"
                />
              );
            })}
          </g>
        </svg>
        <div
          className="x-axis"
          style={{
            marginLeft: margin.left + view.yTickWidth,
            width: view.width,
            height: view.xTickHeight,
          }}
        >
          {view.xTicks.map((xTick: any) => {
            return (
              <span
                title={xTick.tick}
                key={xTick.x}
                onClick={() => {
                  _onColClick(xTick.dimensionsData);
                }}
                style={{
                  left: (view.x(xTick.x) as number) + view.rectWidth / 2,
                  width: view.xTickHeight,
                }}
              >
                {xTick.tick}
              </span>
            );
          })}
        </div>
        <div className="x-label" style={{marginTop: X_LEGEND_MARGIN_TOP}}>
          {xDimension.value}
        </div>
        <div className="y-axis" style={{top: margin.top, left: margin.left - 10}}>
          {view.yTicks.map((yTick: any, index: number) => {
            return (
              <span
                title={yTick.tick}
                style={{
                  width: view.yTickWidth,
                  height: view.rectHeight,
                  lineHeight: view.rectHeight + 'px',
                }}
                onClick={() => {
                  _onRowClick(yTick.dimensionsData);
                }}
                key={index}
              >
                {yTick.tick}
              </span>
            );
          })}
        </div>
        <div className="y-label">{yDimension.value}</div>
      </div>
      <GradientRuler {...props} />
    </div>
  );
};

export default HeatChart;
