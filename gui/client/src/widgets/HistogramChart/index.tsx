import React, {FC, useEffect, useRef, useContext, useMemo} from 'react';
import {useTheme} from '@material-ui/core/styles';
import {scaleTime, scaleLinear, select, axisBottom, axisLeft} from 'd3';
import {default_duration} from '../../utils/Animate';
import {
  legendItemsGetter,
  yDomainGetter,
  compeleteDataGetter,
  dimensionGetter,
  xDomainGetter,
  measureGetter,
  dimensionTypeGetter,
  yAxisFormatterGetter,
  parseDataToXDomain,
  typeEqGetter,
} from '../../utils/WidgetHelpers';
import {cloneObj} from '../../utils/Helpers';
import {widgetFiltersGetter} from '../../utils/Filters';
import {COLUMN_TYPE, MIN_TICK_HEIGHT, MAX_YTICK_NUM} from '../../utils/Consts';
import {formatterGetter} from '../../utils/Formatters';
import {EXTRACT_INPUT_OPTIONS} from '../../utils/Time';
import {color, UNSELECTED_COLOR} from '../../utils/Colors';

import {DEFAULT_WIDGET_WRAPPER_WIDTH, DEFAULT_WIDGET_WRAPPER_HEIGHT} from '../../utils/Layout';
import {I18nContext} from '../../contexts/I18nContext';
import MouseGrp from './HistogramChartMouseGrp';
import Legend from './HistogramLegend';
import {HistogramChartProps} from './types';
import './style.scss';

const margin = {top: 40, right: 20, bottom: 30, left: 80};
const xLabelHeight = 20;
const xLabelMarginTop = 20;
const BAR_GAP = 10;

const HistogramChart: FC<HistogramChartProps> = props => {
  const theme = useTheme();
  const {nls} = useContext(I18nContext);
  const xAxisContainer = useRef<SVGGElement>(null);
  const yAxisContainer = useRef<SVGGElement>(null);
  const yGridLine = useRef<SVGGElement>(null);
  const histogramContainer = useRef<SVGGElement>(null);

  const {
    children: rangeChart,
    config,
    data,
    dataMeta,
    linkMeta,
    wrapperWidth = DEFAULT_WIDGET_WRAPPER_WIDTH,
    wrapperHeight = DEFAULT_WIDGET_WRAPPER_HEIGHT,
    isRange,
    chartHeightRatio = 2 / 3,
    onRangeChange,
    showXLabel,
  } = props;

  const {colorItems = [], measures = [], isShowRange} = config;
  const xDimension = dimensionGetter(config, 'x')!;

  const viewEffect = [
    dataMeta && dataMeta.loading,
    linkMeta && linkMeta.loading,
    wrapperHeight,
    wrapperWidth,
  ];
  const views = useMemo(() => {
    // calculate svg width and height
    const allSvgHeight = wrapperHeight - xLabelHeight - xLabelMarginTop;
    const svgWidth = wrapperWidth;
    const svgHeight = isShowRange ? Math.floor(allSvgHeight * chartHeightRatio) : allSvgHeight;
    const width = svgWidth - margin.left - margin.right,
      height = svgHeight - margin.top - margin.bottom;

    // get x dimension, get xType, get xAxis scale, y measure
    const colorDimension = dimensionGetter(config, 'color');
    const yMeasure = measureGetter(config, 'y') || measures[0];
    const xType = dimensionTypeGetter(xDimension);
    const isTimeChart: boolean = xType === COLUMN_TYPE.DATE;
    const eq = typeEqGetter(xType);

    const xTime = scaleTime().range([0, width]);
    const xLinear = scaleLinear().rangeRound([0, width]);

    // x,y Label
    const yLabels = measures
      .map((m: any) => (m.isRecords ? nls.label_widgetEditor_recordOpt_label_measure : m.label))
      .join(' , ');
    const xLabel = xDimension.label;

    // legendItems
    const legendItems = legendItemsGetter(config).map((item: any) =>
      item.isRecords ? {...item, legendLabel: nls.label_widgetEditor_recordOpt_label_measure} : item
    );
    const compeleteData = compeleteDataGetter(data, xDimension, yMeasure, isTimeChart);
    // compeleteData is already sort by x
    // if we has color dimension, the higher rect need to subtract the lower rect height.
    // so we also need sort by y when x is same
    compeleteData.sort((a: any, b: any) => {
      if (eq(a.x, b.x)) {
        const numB = Number(b.y);
        const numA = Number(a.y);
        if (numA < 0 && numB < 0) {
          return numB - numA;
        }
        return numA - numB;
      }
      return 0;
    });
    // TODO: Time chart is  not compelete data yet.
    const DATA_LEN = compeleteData.reduce((pre: number, cur: any, index: number) => {
      if (!eq(cur.x, compeleteData[index - 1] && compeleteData[index - 1].x)) {
        pre++;
      }
      return pre;
    }, 0);

    const BAR_WIDTH = DATA_LEN ? Math.floor((width - (DATA_LEN - 1) * BAR_GAP) / DATA_LEN) : 0;
    const xDomainExpr = !isRange && config.selfFilter.range && config.selfFilter.range.expr;
    // get scale
    const x = isTimeChart ? xTime : xLinear;
    const y = scaleLinear().range([height, 0]);
    // get xAxis formatter yAxis formatter
    const xAxisFormatter = formatterGetter(xDimension, 'axis');

    const yAxisFormatter = yAxisFormatterGetter(yMeasure, y);
    // calculate x y fill

    const parseData = parseDataToXDomain(compeleteData, xDimension, xDomainExpr);
    const preDataGetter = (datas: any[]) => {
      const filters = widgetFiltersGetter(config);
      return datas.map((item: any) => {
        // add fillColor
        let fill: any;
        const isSelected =
          filters.length === 0 // mean nothing selected eq to all selected
            ? true
            : filters.some((filter: any) => {
                const {expr = {}} = filter;
                let positionX = cloneObj(item[xDimension.as]);
                let {left: exprLeft, right: exprRight} = cloneObj(expr);
                const isDateX = isTimeChart && !xDimension.extract;
                if (isDateX) {
                  positionX = new Date(positionX).getTime();
                  exprLeft = new Date(exprLeft).getTime();
                  exprRight = new Date(exprRight).getTime();
                  return positionX >= exprLeft && positionX <= exprRight;
                }

                return positionX >= exprLeft && positionX <= exprRight;
              });

        if (!isSelected) {
          fill = UNSELECTED_COLOR;
          return {
            x: isTimeChart ? new Date(item[xDimension.as]) : item[xDimension.as],
            y: item[yMeasure.as],
            fill,
          };
        }
        const target = colorDimension
          ? colorItems.find((s: any) => s.as === item.color)
          : colorItems.find((s: any) => s.as === 'y');
        fill = target ? target.color : theme.palette.background.default;
        return {
          x: isTimeChart ? new Date(item[xDimension.as]) : item[xDimension.as],
          y: item[yMeasure.as],
          fill,
        };
      });
    };
    const preData = preDataGetter(parseData);
    // yDomain
    const seriesData = {
      [yMeasure.as]: JSON.parse(JSON.stringify(preData)),
    };

    // xDomain
    let xDomain = xDomainGetter(xDimension, xDomainExpr);

    if (
      xDomain[0] instanceof Date &&
      compeleteData[0] &&
      xDomain[0] > new Date(compeleteData[0][xDimension.as])
    ) {
      xDomain[0] = new Date(compeleteData[0][xDimension.as]);
    }
    const yDomain = yDomainGetter(seriesData, y);
    const isBetweenNgPo = yDomain[0] < 0 && yDomain[1] > 0;

    // update domain
    x.domain(xDomain);
    y.domain(yDomain);

    let id = 1; // for map key
    const renderData: any = [];

    // caculate chart height width for histogram rect render
    preData.forEach((p: any, index: number) => {
      const val = Number(p.y);

      const positionX = x(p.x) <= 0 ? 1 : x(p.x);
      // if yAxis has both negative and positive number.
      // we need to find the zero point as the begin point
      const positionY = isBetweenNgPo ? (val < 0 ? y(0) : y(val)) : val < 0 ? 0 : y(val);
      const barWidth = BAR_WIDTH < 1 ? 1 : BAR_WIDTH;
      let barHeight = !val
        ? 0
        : isBetweenNgPo
        ? val < 0
          ? y(val) - y(0) // because it's start from zero
          : y(0) - y(val) //  end at zero
        : height - y(val);
      const samePositionX = renderData.filter((o: any, i: number) => eq(o.x, p.x) && i < index);
      samePositionX.forEach((d: any) => {
        if ((d.y >= 0 && p.y <= 0) || (d.y <= 0 && p.y >= 0)) {
          // barheight is itself
          return;
        } else {
          barHeight = barHeight - d.barHeight;
        }
      });
      renderData.push({
        ...p,
        barHeight,
        positionY,
        positionX,
        barWidth,
        id,
      });
      id++;
    });
    return {
      svgHeight,
      svgWidth,
      width,
      height,
      yLabels,
      xLabel,
      legendItems,
      xAxisFormatter,
      yAxisFormatter,
      renderData,
      barWidth: BAR_WIDTH,
      xDomain,
      x,
      y,
    };
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(viewEffect)]);

  // animation
  const effectFactors = [
    dataMeta && dataMeta.loading,
    linkMeta && linkMeta.loading,
    wrapperHeight,
    wrapperWidth,
    views.svgHeight,
    JSON.stringify(views.renderData),
  ];
  // console.log(isRange, renderData);
  useEffect(() => {
    if (!dataMeta || dataMeta.loading || !histogramContainer.current) {
      return;
    }
    const xAxis: any = axisBottom(views.x);
    const yAxis: any = axisLeft(views.y);
    xAxis.tickFormat(views.xAxisFormatter);
    yAxis.tickFormat(views.yAxisFormatter);
    yAxis.ticks(
      Math.floor(views.height / MIN_TICK_HEIGHT) > MAX_YTICK_NUM
        ? MAX_YTICK_NUM
        : Math.floor(views.height / MIN_TICK_HEIGHT)
    );
    xAxis.ticks(Math.floor(views.width / 100));
    if (xDimension.extract && xDimension.isBinned) {
      let opt =
        EXTRACT_INPUT_OPTIONS.filter((item: any) => item.value === xDimension.timeBin)[0] || {};
      let numTicks = opt.max || 100;
      xAxis.ticks(numTicks);
    }
    // axis
    const gElArr: any = Array.from(histogramContainer.current.children || []);
    gElArr.forEach((v: any) => {
      const data = views.renderData.find((o: any) => parseInt(v.dataset.id) === o.id);
      if (data) {
        const {barHeight} = data;
        select(v)
          .select('rect')
          .transition()
          .duration(default_duration)
          .attr('height', barHeight);
      }
    });

    // update axis
    select(xAxisContainer.current)
      .transition()
      .call(xAxis);

    select(yAxisContainer.current)
      .transition()
      .call(yAxis);

    select(yGridLine.current).call(
      yAxis
        .tickSizeInner(-views.width)
        .tickSizeOuter(0)
        .tickFormat('')
    );
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(effectFactors)]);

  // tooltipContentGetter
  const tooltipContentGetter = (tooltipData: any) => {
    return (
      <ul>
        {tooltipData.data
          .filter((d: any) => d)
          .map((d: any, index: number) => {
            const fill = d.fill || '#fff';
            const formatValue = views.yAxisFormatter(d.y);
            return (
              <li key={index}>
                <span
                  className="mark"
                  style={{
                    background: fill || color(d.as),
                  }}
                />
                {d.color} {formatValue}
              </li>
            );
          })}
      </ul>
    );
  };

  const tooltipTitleGetter = (tooltipData: any) => {
    const xDomainVal =
      tooltipData.data && tooltipData.data.length
        ? tooltipData.data[0] && tooltipData.data[0].x
        : nls.label_StackedBarChart_nodata;
    return <div>{views.xAxisFormatter(xDomainVal)}</div>;
  };
  return (
    <div
      style={{
        width: wrapperWidth,
        height: wrapperHeight,
        background: theme.palette.background.paper,
      }}
      className={`histogram-chart`}
    >
      <svg width={views.svgWidth} height={views.svgHeight}>
        {/* xAxis yAxis */}
        <g
          width={views.width}
          height={views.height}
          transform={`translate(${margin.left}, ${margin.top})`}
        >
          <g className="axis axis--x" pointerEvents="none">
            <g className="grid-line" transform={`translate(0,${views.height})`}></g>
            <g ref={xAxisContainer} transform={`translate(0, ${views.height})`} />
          </g>
          <g className="axis axis--y" pointerEvents="none">
            <g className="grid-line" ref={yGridLine} />
            <g ref={yAxisContainer} />
            <text
              fill="#000"
              transform="rotate(-90)"
              y={-margin.left + 10}
              x={-10}
              dy="0.71em"
              textAnchor="end"
            >
              {views.yLabels}
            </text>
          </g>
        </g>

        {/* histogram chart */}
        <g
          width={views.width}
          height={views.height}
          transform={`translate(${margin.left}, ${margin.top})`}
          ref={histogramContainer}
        >
          {views.renderData.map((l: any) => {
            const {barWidth, positionX, positionY, fill, id} = l;

            return (
              <g key={`${l.as}${id}`} className="bar" data-id={id}>
                <rect
                  fill={fill}
                  y={positionY}
                  height={0}
                  width={barWidth || 0}
                  x={positionX || 0}
                ></rect>
              </g>
            );
          })}
        </g>
      </svg>
      {isShowRange ? rangeChart : null}
      {showXLabel && <div className="x-label">{views.xLabel}</div>}

      <MouseGrp
        config={config}
        onRangeChange={onRangeChange}
        dataMeta={dataMeta}
        linkMeta={linkMeta}
        svgHeight={views.svgHeight}
        svgWidth={views.svgWidth}
        margin={margin}
        x={views.x}
        y={views.y}
        isRangeChart={!!isRange}
        legendItems={views.legendItems}
        renderData={views.renderData}
        xDomain={views.xDomain}
        barWidth={views.barWidth}
        tooltipTitleGetter={tooltipTitleGetter}
        tooltipContentGetter={tooltipContentGetter}
      ></MouseGrp>
      {!isRange && <Legend legendData={config.colorItems || []} />}
    </div>
  );
};

export default HistogramChart;
