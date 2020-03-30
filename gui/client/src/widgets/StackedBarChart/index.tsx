import React, {useState, useRef, useEffect, useContext} from 'react';
import {scaleLinear, scaleBand, select, axisBottom, axisLeft} from 'd3';
import {useTheme} from '@material-ui/core/styles';
import {
  stackedBarYDomainGetter,
  dimensionGetter,
  measureGetter,
  yAxisFormatterGetter,
  stackedBarHoverDataGetter,
} from '../../utils/WidgetHelpers';
import {color, UNSELECTED_COLOR} from '../../utils/Colors';
import {formatterGetter} from '../../utils/Formatters';
import {widgetFiltersGetter} from '../../utils/Filters';
import {rootContext} from '../../contexts/RootContext';
import {I18nContext} from '../../contexts/I18nContext';
import {default_duration, animate, getStackedBarTransistData} from '../../utils/Animate';
import {DEFAULT_WIDGET_WRAPPER_WIDTH, DEFAULT_WIDGET_WRAPPER_HEIGHT} from '../../utils/Layout';
import {StackedBarChartProps} from './types';
import './style.scss';

const StackedBarChart = (props: StackedBarChartProps) => {
  const {nls} = useContext(I18nContext);
  const theme = useTheme();
  const {showTooltip, hideTooltip} = useContext(rootContext);
  const {
    config,
    data,
    wrapperWidth = DEFAULT_WIDGET_WRAPPER_WIDTH,
    wrapperHeight = DEFAULT_WIDGET_WRAPPER_HEIGHT,
    onStackedBarClick,
    dataMeta,
  } = props;
  const {stackType = 'vertical', colorItems = []} = config;
  const [showRect, setShowRect] = useState();
  const [preRenderData, setPreRenderData] = useState();
  const margin = {
      top: 40,
      right: 20,
      bottom: 30,
      left: 80,
    },
    width = wrapperWidth - margin.left - margin.right,
    height = wrapperHeight - margin.top - margin.bottom;
  const chartContainer = useRef<any>(null);

  // get width measure
  const xDimension = dimensionGetter(config, 'x')!;
  const yMeasure = measureGetter(config, 'y')!;
  const colorDimension = dimensionGetter(config, 'color');
  const formatter = formatterGetter(yMeasure);
  // setting up domains
  const x = scaleBand().range([0, width]);
  const y = scaleLinear().range([height, 0]);
  const yAxisFormatter = yAxisFormatterGetter(yMeasure, y);

  const xAxis: any = axisBottom(x);
  const yAxis: any = axisLeft(y);
  yAxis.tickFormat(yAxisFormatter);

  data.sort((item1: any, item2: any) => {
    if (item1[xDimension.as] > item2[xDimension.as]) {
      return 1;
    }
    if (item1[xDimension.as] < item2[xDimension.as]) {
      return -1;
    }
    if (!colorDimension) {
      return 1;
    }
    if (item1[colorDimension.as] > item2[colorDimension.as]) {
      return 1;
    }
    return -1;
  });

  const filters = widgetFiltersGetter(config);
  // domain
  const xDomain: any = Array.from(new Set(data.map((item: any) => item.x)));
  const yDomain: any = stackedBarYDomainGetter(data, xDomain, stackType);
  let yHeight: any = scaleLinear().range([height, 0]);
  x.domain(xDomain);
  if (yDomain.min >= 0) {
    y.domain([yDomain.min, yDomain.max]);
    yHeight.domain([yDomain.max, 0]);
  }
  if (yDomain.max <= 0) {
    y.domain([yDomain.min, yDomain.max]);
    yHeight.domain([yDomain.min, 0]);
  }
  if (yDomain.min < 0 && yDomain.max > 0) {
    y.domain([yDomain.min, yDomain.max]);
    yHeight.domain([yDomain.max - yDomain.min, 0]);
  }

  // compute layout properties
  const BAR_GAP = 10,
    _BAR_GAP = 1,
    barWidth = (width - (xDomain.length - 1) * BAR_GAP) / xDomain.length,
    fontSize = 12;

  let baseHeight = height,
    accumulateHeight = 0,
    accumulateTopHeight = 0,
    accumulateBottomHeight = 0,
    lastXAxisValue = '',
    yMin = yDomain.min,
    yMax = yDomain.max;

  const _addLayout = (datas: any[]) => {
    return 'vertical' ? _addVerticalLayout(datas) : _addHorizontalLayout(datas);
  };

  const _addVerticalLayout = (datas: any[]) => {
    return datas.map((item: any) => {
      let gheight = yHeight(item[yMeasure.as]);
      const isEqualToLast = lastXAxisValue === item[xDimension.as];
      item.setting = {
        x: x(item[xDimension.as]),
        width: barWidth,
        height: gheight,
      };
      if (yMin >= 0) {
        item.setting.y = baseHeight - gheight - (isEqualToLast ? accumulateHeight : 0);
      }
      if (yMax <= 0) {
        item.setting.y = isEqualToLast ? accumulateHeight : 0;
      }
      // case yMin === yMax
      if (yMin === yMax) {
        gheight = 1;
        item.setting.y = height / 2;
        item.setting.height = gheight;
      }
      if (yMin < 0 && yMax > 0) {
        accumulateTopHeight = isEqualToLast ? accumulateHeight : 0;
        accumulateBottomHeight = isEqualToLast ? accumulateBottomHeight : 0;
        const y0 = y(0);
        // cal gHeight
        gheight = yHeight(Math.abs(item[yMeasure.as]));
        // cal y
        item.setting.y =
          item[yMeasure.as] >= 0
            ? y0 - gheight - (isEqualToLast ? accumulateTopHeight : 0)
            : y0 + (isEqualToLast ? accumulateBottomHeight : 0);
        item.setting.height = gheight;
        item[yMeasure.as] >= 0 && (accumulateTopHeight += gheight);
        item[yMeasure.as] < 0 && (accumulateBottomHeight += gheight);
      }
      accumulateHeight = isEqualToLast ? accumulateHeight + gheight : gheight;
      lastXAxisValue = item[xDimension.as];
      return item;
    });
  };
  const _addHorizontalLayout = (datas: any[]) => {
    return datas.map((item: any) => {
      let _group = datas.filter((d: any) => d[xDimension.as] === item[xDimension.as]),
        groupLen = _group.length,
        width = barWidth / groupLen - _BAR_GAP,
        sortIndex = _group.findIndex((_g: any) => JSON.stringify(_g) === JSON.stringify(item)),
        startPos = x(item[xDimension.as]) || 0,
        gheight = yHeight(item[yMeasure.as]);

      item.setting = {
        x: startPos + (width + _BAR_GAP) * sortIndex,
        width: width,
        height: gheight,
      };
      if (yMin >= 0) {
        item.setting.y = baseHeight - gheight;
      }
      if (yMax <= 0) {
        item.setting.y = 0;
      }
      if (yMin < 0 && yMax > 0) {
        gheight = yHeight(Math.abs(item[yMeasure.as]));
        const y0 = y(0);
        item.setting.y = item[yMeasure.as] >= 0 ? y0 - gheight : y0;
        item.setting.height = gheight;
      }
      if (yMin === yMax) {
        gheight = 1;
        item.setting.height = gheight;
        item.setting.y = y(0);
      }
      return item;
    });
  };
  const _addColor = (datas: any[]) => {
    return datas.map((item: any) => {
      let fill: any;
      const allSelected = filters.length === 0;
      const targetSelected = filters.some((filter: any) => {
        const {expr = {}} = filter;
        return expr.right === item[xDimension.as];
      });
      const isSelected = allSelected || targetSelected;
      if (!isSelected) {
        fill = UNSELECTED_COLOR;
        return {
          ...item,
          setting: {
            ...item.setting,
            fill,
          },
        };
      }
      const target = colorItems.find((s: any) => s.as === (colorDimension ? item.color : 'y'));
      fill = target ? target.color : theme.palette.background.default;
      return {
        ...item,
        setting: {
          ...item.setting,
          fill,
        },
      };
    });
  };
  const getRenderData = (datas: any[]) => {
    return _addColor(_addLayout(datas));
  };

  const renderData = getRenderData(data);
  // tooltip
  const tooltipTitleGetter = (hoverDatas: any) => {
    const xDomainVal = hoverDatas.length > 0 ? hoverDatas[0].x : nls.label_StackedBarChart_nodata;
    return <>{xDomainVal}</>;
  };

  const tooltipContentGetter = (hoverDatas: any) => {
    return (
      <ul
        style={{
          listStyleType: 'none',
          padding: 0,
        }}
      >
        {hoverDatas.map((d: any = {}, index: number) => {
          const fill = d.fill || '#fff';
          return (
            <li key={index}>
              <span
                className="mark"
                style={{
                  background: fill || color(d.as),
                }}
              />
              {d.color} {formatter(d.y)}
            </li>
          );
        })}
      </ul>
    );
  };

  const _onMouseMove = (event: any, domain: any) => {
    const hoverData = stackedBarHoverDataGetter(renderData).filter((d: any) => d.x === domain);
    const position = {event};
    showTooltip({
      tooltipData: hoverData,
      position,
      titleGetter: tooltipTitleGetter,
      contentGetter: tooltipContentGetter,
    });
  };

  const _onMouseEnter = (domain: any) => {
    setShowRect(domain);
  };

  const _onMouseLeave = () => {
    hideTooltip();
    setShowRect(null);
  };

  const effectFactors = JSON.stringify([
    dataMeta && dataMeta.loading,
    wrapperWidth,
    wrapperHeight,
    stackType,
    config.filter,
    config.colorItems,
    renderData.length,
  ]);
  // animation
  useEffect(() => {
    if (!preRenderData || preRenderData.length === 0) {
      setPreRenderData(renderData);
      return;
    }
    let ts: any,
      duration: number = default_duration;

    if (ts) {
      ts.stop();
    }
    ts = animate({
      duration,
      curve: [0, duration],
      onAnimate: function(elapsed: any) {
        setPreRenderData(
          getStackedBarTransistData(preRenderData, renderData, elapsed, default_duration)
        );
      },
    });
    ts.start();
    return () => {
      ts.stop();
    };
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectFactors]);

  return (
    <div
      className="z-chart z-stackedbar-chart"
      ref={chartContainer}
      style={{
        background: theme.palette.background.paper,
      }}
    >
      <div
        className="svg-wrapper"
        style={{
          height: wrapperHeight,
          width: wrapperWidth,
          overflow: 'hidden',
          fontSize: `${fontSize}px`,
          position: 'relative',
        }}
      >
        <svg width={wrapperWidth} height={wrapperHeight}>
          {/* xAxis, yAxis */}
          <g width={width} height={height} transform={`translate(${margin.left}, ${margin.top})`}>
            <g className="axis axis--x" pointerEvents="none">
              <g transform={`translate(0, ${height})`} ref={node => select(node).call(xAxis)} />
            </g>
            <g className="axis axis--y" pointerEvents="none">
              <g ref={node => select(node).call(yAxis)} />
              <g
                className="grid-line"
                ref={node =>
                  select(node).call(
                    yAxis
                      .tickSizeInner(-width)
                      .tickSizeOuter(0)
                      .tickFormat('')
                  )
                }
              />
              <text
                fill="#FFF"
                transform="rotate(-90)"
                y={-margin.left + 10}
                x={-10}
                dy="0.71em"
                textAnchor="end"
              >
                {`${nls[`label_widgetEditor_expression_${yMeasure.expression}`]}  ${
                  yMeasure.value
                }`}
              </text>
            </g>
          </g>
          {/* toolBar */}
          <g width={width} height={height} transform={`translate(${margin.left}, ${margin.top})`}>
            {xDomain.map((domain: any, index: number) => {
              return (
                <rect
                  key={`${domain}${index}`}
                  x={x(domain)}
                  y={0}
                  width={barWidth}
                  height={height}
                  fill={'transparent'}
                  onMouseEnter={() => {
                    _onMouseEnter(domain);
                  }}
                  onMouseMove={e => {
                    _onMouseMove(e, domain);
                  }}
                  onMouseLeave={() => {
                    _onMouseLeave();
                  }}
                />
              );
            })}
          </g>
          {/* rects */}
          <g width={width} height={height} transform={`translate(${margin.left}, ${margin.top})`}>
            {(preRenderData || renderData).map((d: any, i: number) => {
              const {fill, x, y, width, height} = d.setting;
              return (
                <g
                  key={i}
                  fill={fill}
                  onClick={() => {
                    onStackedBarClick(d);
                  }}
                  onMouseMove={e => {
                    _onMouseMove(e, d.x);
                  }}
                  onMouseLeave={_onMouseLeave}
                  className="row"
                >
                  <rect x={x} y={y} width={width} height={height} />
                </g>
              );
            })}
          </g>
          {/* vertical line */}
          <g width={width} height={height} transform={`translate(${margin.left}, ${margin.top})`}>
            {xDomain.map((domain: any, index: number) => {
              return (
                <line
                  key={`${domain}${index}`}
                  y1={0}
                  y2={height}
                  x1={(x(domain) || 0) + barWidth / 2}
                  x2={(x(domain) || 0) + barWidth / 2}
                  stroke="#fff"
                  style={{
                    display: showRect === domain ? '' : 'none',
                    pointerEvents: 'none',
                  }}
                />
              );
            })}
          </g>
        </svg>
      </div>
    </div>
  );
};

export default StackedBarChart;
