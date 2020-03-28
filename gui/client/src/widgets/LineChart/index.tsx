import React, {FC, Fragment, useMemo, useState, useEffect, useRef, useContext} from 'react';
import {
  scaleTime,
  scaleLinear,
  select,
  axisBottom,
  axisLeft,
  line as d3Line,
  area as d3Area,
} from 'd3';
import {I18nContext} from '../../contexts/I18nContext';
import Bin from './Bin';
import TimeSelector from './TimeSelector';
import Legend from '../common/Legend';
import MouseGrp from './MouseGrp';
import {LineChartProps} from './types';
import {COLUMN_TYPE, MIN_TICK_HEIGHT, MAX_YTICK_NUM} from '../../utils/Consts';
import {color} from '../../utils/Colors';
import {EXTRACT_INPUT_OPTIONS} from '../../utils/Time';
import {
  seriesDataGetter,
  legendItemsGetter,
  yDomainGetter,
  lineChartBinDataGetter,
  timeSelectorDataGetter,
  legendDataGetter,
  dimensionGetter,
  xDomainGetter,
  measureGetter,
  dimensionTypeGetter,
  yAxisFormatterGetter,
  parseDataToXDomain,
} from '../../utils/WidgetHelpers';
import {formatterGetter} from '../../utils/Formatters';
import {DEFAULT_WIDGET_WRAPPER_WIDTH, DEFAULT_WIDGET_WRAPPER_HEIGHT} from '../../utils/Layout';
import './style.scss';

const {interpolatePath} = require('d3-interpolate-path');

// LINECHART realted consts
const cssClassRegexp: any = new RegExp(/[.//:()$\s+@&#?^/!,~/*%'[\]]/g);
const margin = {top: 40, right: 20, bottom: 30, left: 80};
const xLabelHeight = 20;
const xLabelMarginTop = 20;
export const AREA_OPACITY = 0.67;
// assure data should not be empty
// empty data should be checked on parent side
const LineChart: FC<LineChartProps> = props => {
  const {nls} = useContext(I18nContext);
  const xAxisContainer = useRef<SVGGElement>(null);
  const yAxisContainer = useRef<SVGGElement>(null);
  const xGridLine = useRef<SVGGElement>(null);
  const yGridLine = useRef<SVGGElement>(null);
  const {
    children: rangeChart,
    isRange,
    config,
    chartHeightRatio = 1,
    data,
    dataMeta,
    wrapperWidth = DEFAULT_WIDGET_WRAPPER_WIDTH,
    wrapperHeight = DEFAULT_WIDGET_WRAPPER_HEIGHT,
  } = props;
  const clipId = `chart-${config.id}`;
  const allSvgHeight = wrapperHeight - xLabelHeight - xLabelMarginTop;
  const isShowRange = config.isShowRange;
  const svgWidth = wrapperWidth;
  const svgHeight = isShowRange ? Math.floor(allSvgHeight * chartHeightRatio) : allSvgHeight;

  const {measures = []} = config;
  const width = svgWidth - margin.left - margin.right,
    height = svgHeight - margin.top - margin.bottom;

  const xTime = scaleTime().range([0, width]);
  const xLinear = scaleLinear().range([0, width]);

  // get x dimension, get xType, get xAxis scale
  const xDimension = dimensionGetter(config, 'x')!;
  const colorDimension = dimensionGetter(config, 'color');
  const xType = dimensionTypeGetter(xDimension);
  const isTimeChart: boolean = xType === COLUMN_TYPE.DATE;

  // get scale
  const x = isTimeChart ? xTime : xLinear;
  const y = scaleLinear().range([height, 0]);
  // create line builder
  const line = config.isArea
    ? d3Area()
        .x((d: any) => x(d.x))
        .y0(y(0))
        .y1((d: any) => y(d.y))
    : d3Line()
        .x((d: any) => x(d.x))
        .y((d: any) => y(d.y));

  // legendItems, yLabels, seriesData
  const xLabel = xDimension.label;
  const legendItems = useMemo(
    () =>
      legendItemsGetter(config).map((item: any) =>
        item.isRecords
          ? {...item, legendLabel: nls.label_widgetEditor_recordOpt_label_measure}
          : item
      ),
    [config, nls.label_widgetEditor_recordOpt_label_measure]
  );
  const isByColorDimension = !!colorDimension;
  const yLabels = measures
    .map((m: any) => (m.isRecords ? nls.label_widgetEditor_recordOpt_label_measure : m.label))
    .join(' , ');

  const _data = useMemo(() => parseDataToXDomain(data, xDimension), [data, xDimension]);
  const seriesData = useMemo(
    () => seriesDataGetter(legendItems, _data, xDimension.as, isTimeChart, isByColorDimension),
    [isByColorDimension, _data, legendItems, isTimeChart, xDimension.as]
  );

  // xDomain, yDomain
  const xDomainExpr = config.selfFilter.range && config.selfFilter.range.expr;
  let xDomain = xDomainGetter(xDimension, xDomainExpr, isRange);
  // domain correction
  if (isTimeChart && _data[0] && xDomain[0] > new Date(_data[0][xDimension.as])) {
    xDomain[0] = new Date(_data[0][xDimension.as]);
  }
  const yDomain = yDomainGetter(seriesData, y);
  // update domain
  x.domain(xDomain);
  y.domain(yDomain);

  // axis
  const xAxis: any = axisBottom(x);
  const yAxis: any = axisLeft(y);
  const xAxisFormatter = formatterGetter(xDimension, 'axis');
  // get yAxis formatter
  // TODO: when LineChart has multy Measures, which format should choose for yAxis ???  currently, we select the first measure's format as yAxis's format as MapD
  const yMeasure = measureGetter(config, 'y') || config.measures[0];
  const yAxisFormatter = yAxisFormatterGetter(yMeasure, y);

  // animation
  const linesContainer = useRef<any>(null);
  const effectFactors = [
    dataMeta && dataMeta.loading,
    wrapperHeight,
    wrapperWidth,
    config.dimensions,
    config.measures,
  ];

  useEffect(() => {
    if (!dataMeta || dataMeta.loading) {
      return;
    }
    let pathLegendItems: any[] = [],
      pointLegendItems: any = [];

    legendItems.forEach((item: any) => {
      seriesData[item.as].length < 2 ? pointLegendItems.push(item) : pathLegendItems.push(item);
    });

    pointLegendItems.forEach((item: any) => {
      let pointClass = `.circleclass_${item.as.replace(cssClassRegexp, '_')}`;

      if (seriesData[item.as][0]) {
        select(linesContainer.current)
          .select(pointClass)
          .transition()
          .attr('r', 3)
          .attr('cx', x(seriesData[item.as][0].x))
          .attr('cy', y(seriesData[item.as][0].y));
      }
    });

    pathLegendItems.forEach((item: any) => {
      const datas = seriesData[item.as];
      const newPath = line(datas);
      const lineNodeClass = `.class_${item.as.replace(cssClassRegexp, '_')}`;
      const cNodeClass = `.circleclass_${item.as.replace(cssClassRegexp, '_')}`;

      select(linesContainer.current)
        .select(cNodeClass)
        .transition()
        .attr('r', 0)
        .attr('cx', -100)
        .attr('cy', -100);

      select(linesContainer.current)
        .select(lineNodeClass)
        .transition()
        .attrTween('d', function() {
          const previous = select(this).attr('d');
          return interpolatePath(previous, newPath || null);
        });
    });

    if (pathLegendItems.length === 0) {
      select(linesContainer.current)
        .selectAll('path')
        .transition()
        .attrTween('d', function() {
          return interpolatePath(null, null);
        });
    }

    // update axis
    select(xAxisContainer.current)
      .transition()
      .call(xAxis);

    select(yAxisContainer.current)
      .transition()
      .call(yAxis);

    select(xGridLine.current).call(
      xAxis
        .tickSizeInner(-height)
        .tickSizeOuter(0)
        .tickFormat('')
    );

    select(yGridLine.current).call(
      yAxis
        .tickSizeInner(-width)
        .tickSizeOuter(0)
        .tickFormat('')
    );

    return () => {};
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(effectFactors)]);
  // compute is show or bin
  const isBinned = config.dimensions.some((d: any) => d.isBinned);
  const isExtract = config.dimensions.some((d: any) => d.extract);
  const showBin = isBinned && !isExtract && isTimeChart;

  // get timeSelectorData data
  const timeSelectorData = timeSelectorDataGetter(config, x, showBin);
  // get legend data
  const legendData = legendDataGetter(config, legendItems);
  // console.log("====legendData is : ", legendData);
  const binData = lineChartBinDataGetter(config, showBin);

  // tooltipContentGetter
  const tooltipContentGetter = (tooltipData: any) => {
    return (
      <ul>
        {tooltipData.data
          .filter((d: any) => d)
          .map((d: any) => {
            const _color = d.color;
            return (
              <li className="" key={d.as}>
                <span className="mark" style={{background: _color || color(d.as)}} />
                {d.value} {d.formattedValue}
              </li>
            );
          })}
      </ul>
    );
  };

  const tooltipTitleGetter = (tooltipData: any) => {
    return <>{`${xAxisFormatter(tooltipData.xV) || nls.label_StackedBarChart_nodata}`}</>;
  };

  const isExtractBin = xDimension.extract && xDimension.isBinned;
  let xTickNums = Math.floor(width / 100);
  if (isExtractBin) {
    let opt =
      EXTRACT_INPUT_OPTIONS.filter((item: any) => item.value === xDimension.timeBin)[0] || {};
    xTickNums = opt.max || 100;
  }
  xAxis.ticks(xTickNums);
  const yTickNums =
    Math.floor(height / MIN_TICK_HEIGHT) > MAX_YTICK_NUM
      ? MAX_YTICK_NUM
      : Math.floor(height / MIN_TICK_HEIGHT);

  xAxis.tickFormat(xAxisFormatter);
  yAxis.tickFormat(yAxisFormatter);
  yAxis.ticks(yTickNums);

  const [timeSelection, setTimeSelection] = useState<any>([]);
  return (
    <div
      className={`z-chart z-line-chart ${clipId}`}
      style={{width: wrapperWidth, height: wrapperHeight}}
    >
      <svg width={svgWidth} height={svgHeight}>
        <g width={width} height={height} transform={`translate(${margin.left}, ${margin.top})`}>
          <g className="axis axis--x" pointerEvents="none">
            <g className="grid-line" transform={`translate(0, ${height})`} ref={xGridLine} />
            <g ref={xAxisContainer} transform={`translate(0, ${height})`} />
          </g>
          <g className="axis axis--y" pointerEvents="none">
            <g className="grid-line" ref={yGridLine} />
            <g ref={yAxisContainer} />
            <text
              fill="#FFF"
              transform="rotate(-90)"
              y={-margin.left + 10}
              x={-10}
              dy="0.71em"
              textAnchor="end"
            >
              {!isRange && yLabels}
            </text>
          </g>
        </g>
        <g
          width={width}
          height={height}
          transform={`translate(${margin.left}, ${margin.top})`}
          clipPath={`url(#${clipId})`}
        >
          <g className="lines" ref={linesContainer}>
            {legendItems
              .filter((i: any) => seriesData[i.as].length)
              .map((l: any, index: number) => {
                return (
                  <Fragment key={`${l.as}${index}`}>
                    <path
                      className={`line class_${l.as.replace(cssClassRegexp, '_')}`}
                      data-as={l.as}
                      d=""
                      fill={config.isArea ? l.color : 'none'}
                      stroke={l.color || color(l.as)}
                    />
                    <circle
                      className={`circleclass_${l.as.replace(cssClassRegexp, '_')}`}
                      data-as={l.as}
                      key={index}
                      cy={-100}
                      cx={-100}
                      r={3}
                      stroke={l.color || color(l.as)}
                      fill={l.color || color(l.as)}
                    />
                    {config.isArea && (
                      <path
                        className={`line class_${l.as.replace(cssClassRegexp, '_')}`}
                        data-as={l.as}
                        d=""
                        opacity={config.isArea ? AREA_OPACITY : 1}
                        fill={l.color || color(l.as)}
                        stroke={l.color || color(l.as)}
                      />
                    )}
                  </Fragment>
                );
              })}
          </g>
        </g>
        <defs>
          <clipPath id={clipId}>
            <rect width={svgWidth} height={svgHeight} transform="translate(-0, -0)" />
          </clipPath>
        </defs>
      </svg>
      {isShowRange && rangeChart}
      {(!isShowRange || isRange) && <div className="x-label">{xLabel}</div>}
      <MouseGrp
        {...props}
        x={x}
        y={y}
        svgHeight={svgHeight}
        svgWidth={svgWidth}
        margin={margin}
        width={width}
        height={height}
        legendItems={legendItems}
        tooltipContentGetter={tooltipContentGetter}
        tooltipTitleGetter={tooltipTitleGetter}
        seriesData={seriesData}
        xDimension={xDimension}
        isTimeChart={isTimeChart}
        xDomain={xDomain}
        setTimeSelection={setTimeSelection}
      />
      <Bin binData={binData} />
      <TimeSelector timeSelectorData={timeSelectorData} timeSelection={timeSelection} />
      {!isRange && <Legend legendData={legendData} />}
    </div>
  );
};

export default LineChart;
