import React, {FC, useState, useRef, useEffect} from 'react';
import {useTheme} from '@material-ui/core/styles';
import {scaleLinear, scaleBand, select, max, min, axisBottom} from 'd3';
import GradientRuler from '../common/GradientRuler';
import {BarChartProps, Data} from './types';
import {cloneObj, isValidValue} from '../../utils/Helpers';
import {measureGetter, dimensionDataGetter} from '../../utils/WidgetHelpers';
import {rangeFormatter, formatterGetter} from '../../utils/Formatters';
import {default_duration} from '../../utils/Animate';
import {dimensionsDataToFilterExpr} from '../../utils/Filters';
import {genColorGetter} from '../../utils/Colors';
import {DEFAULT_WIDGET_WRAPPER_WIDTH, DEFAULT_WIDGET_WRAPPER_HEIGHT} from '../../utils/Layout';
import './style.scss';

// Chart related consts
const MIN_BAR_HEIGHT = 16;
const BAR_GAP = 3;
const fontSize = 12;
const margin = {top: 10, right: 30, bottom: 50, left: 20};
const rulerHeight = 30;
const axisHeight = 60;
// bar data preFormat
const barDataPreFormat = (data: Data, filters: any, dimensionFormatter: Function): any[] => {
  return data.map((row: any, i: number) => {
    let newRow: any = cloneObj(row);
    let currentFilterExpr = dimensionsDataToFilterExpr(newRow.dimensionsData);
    let filterNames = Object.keys(filters);

    newRow.data = {};
    newRow.data.filters = [];
    newRow.data.selected =
      filterNames.length === 0
        ? true
        : filterNames.filter((f: any) => {
            if (filters[f].expr === currentFilterExpr) {
              newRow.data.filters.push(f);
              return true;
            } else {
              return false;
            }
          }).length > 0;

    newRow.data.dimensionsData = newRow.dimensionsData;
    newRow.index = i;
    newRow.y = dimensionFormatter(newRow.data.dimensionsData);
    return newRow;
  });
};

const BarChart: FC<BarChartProps> = props => {
  const theme = useTheme();
  const {
    config,
    data,
    dataMeta,
    wrapperWidth = DEFAULT_WIDGET_WRAPPER_WIDTH,
    wrapperHeight = DEFAULT_WIDGET_WRAPPER_HEIGHT,
    onBarClick,
  } = props;

  let getColor = genColorGetter(config);

  const [view, setView] = useState<any>({
    svgWidth: wrapperWidth,
    svgHeight: wrapperHeight,
    width: wrapperWidth - margin.left - margin.right,
    height: wrapperHeight - margin.top - margin.bottom,
    barData: [],
    widthMeasure: {as: ''},
    x: scaleLinear().range([0, wrapperWidth - margin.left - margin.right]),
    y: scaleBand().rangeRound([0, wrapperHeight - margin.top - margin.bottom]),
  });

  const effectFactors = JSON.stringify([
    dataMeta && dataMeta.loading,
    wrapperHeight,
    wrapperWidth,
    config.dimensions,
    config.measures,
    config.filter,
  ]);

  useEffect(() => {
    let _view: any = {
      svgWidth: wrapperWidth,
      svgHeight: wrapperHeight,
      width: wrapperWidth - margin.left - margin.right,
      height: wrapperHeight - margin.top - margin.bottom,
    };
    // setting up domains
    _view.x = scaleLinear().range([0, _view.width]);
    _view.y = scaleBand().rangeRound([0, _view.height]);
    // get width measure
    _view.widthMeasure = measureGetter(config, 'width')!;
    // format pie data
    _view.barData = barDataPreFormat(
      dimensionDataGetter(config.dimensions, data),
      config.filter,
      rangeFormatter
    );
    let maxWidth: number = 0;
    _view.barData.forEach((d: any) => {
      maxWidth = Math.max(maxWidth, d[_view.widthMeasure.as]);
    });

    // measure formatter
    _view.barMeasureFormatter = (v: any) => {
      return formatterGetter(_view.widthMeasure)(v);
    };

    // domain
    const xDomainGetter = (data: any, measure: any) => {
      let _data = data.map((d: any) => d[measure.as]);
      return [(min(_data) || 0) > 0 ? 0 : min(_data), max(_data)];
    };
    const xDomain: any = xDomainGetter(_view.barData, _view.widthMeasure);
    _view.x.domain(xDomain);
    _view.y.domain(_view.barData.map((b: any) => b.y));

    // event
    _view.onClick = (e: any) => {
      let index = e.target.dataset.index * 1;
      onBarClick && onBarClick(_view.barData.filter((bar: any) => bar.index === index)[0]);
    };

    _view.barHeight = _view.y.bandwidth() > MIN_BAR_HEIGHT ? _view.y.bandwidth() : MIN_BAR_HEIGHT;
    _view.allBarHeight =
      _view.barData.length * _view.barHeight +
      margin.top +
      margin.bottom +
      (_view.barData.length - 1) * BAR_GAP;
    _view.dy = _view.barHeight / 2 + 2;

    // update
    setView(_view);

    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectFactors]);

  // animation
  useEffect(() => {
    const xAxis: any = axisBottom(view.x);

    // use d3 to do the transition, react handle the dom part
    const gElArr: any = Array.from(barContainer.current!.children || []);
    gElArr.forEach((v: any) => {
      const newRectRow = view.barData.find((d: any) => d.y === v.dataset.y) || {};
      const newWidth = view.x(newRectRow[view.widthMeasure.as] || 0);

      select(v)
        .select('rect')
        .transition()
        .duration(default_duration)
        .attr('width', newWidth);
    });

    // can not put this in jsx , may be transition conflict with react dom update
    select(xAxisContainer.current)
      .transition()
      .call(xAxis);

    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(view)]);

  //animation
  const barContainer = useRef<SVGGElement>(null);
  const xAxisContainer = useRef<SVGGElement>(null);

  return (
    <div
      className="z-chart z-bar-chart"
      style={{
        background: theme.palette.background.paper,
      }}
    >
      <div
        className="svg-wrapper"
        style={{
          height: wrapperHeight - rulerHeight - axisHeight,
          overflowY: 'auto',
          overflowX: 'hidden',
          fontSize: `${fontSize}px`,
        }}
      >
        <svg width={view.svgWidth} height={view.allBarHeight}>
          <g
            width={view.width}
            height="100%"
            transform={`translate(${margin.left}, ${margin.top})`}
            ref={barContainer}
          >
            {view.barData.map((d: any, i: number) => {
              const value = d[view.widthMeasure.as] as number;
              const zero = view.x.invert(0);
              const rectWidth = view.x(value);
              const color = getColor(
                isValidValue(d.color) ? d.color : d[config.dimensions[0].as],
                theme.palette.background.default
              );
              return (
                <g
                  className={`${d.data.selected ? '' : 'unselected'}`}
                  key={i}
                  transform={`translate(0, ${i * view.barHeight + BAR_GAP * i})`}
                  onClick={view.onClick}
                  data-y={d.y}
                >
                  <rect
                    className={`row ${d.data.selected ? '' : 'unselected'}`}
                    width={0}
                    height={view.barHeight}
                    data-index={d.index}
                    stroke={color}
                    fill={color}
                    x={view.x(value > zero ? zero : value)}
                  />
                  <text x={view.x(value > zero ? zero : value) + 3} dy={view.dy} fontWeight="bold">
                    {d.y}
                  </text>
                  <text dy={view.dy} fill="#eee" textAnchor="end" x={rectWidth - 20}>
                    {view.barMeasureFormatter(value)}
                  </text>
                </g>
              );
            })}
          </g>
        </svg>
      </div>
      <div>
        <svg width={view.svgWidth} height={axisHeight}>
          <g width={view.width} transform={`translate(${margin.left}, ${margin.top})`}>
            <g className="axis" pointerEvents="none">
              <g transform={`translate(0, ${0})`} ref={xAxisContainer} />
            </g>
          </g>
        </svg>
      </div>
      <GradientRuler {...props} />
    </div>
  );
};

export default BarChart;
