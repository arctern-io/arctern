import React, {useState, useEffect, useContext, Fragment} from 'react';
import {useTheme} from '@material-ui/core/styles';
import {select, axisBottom, axisLeft} from 'd3';
import {rootContext} from '../../contexts/RootContext';
import {I18nContext} from '../../contexts/I18nContext';
import Table from '../../components/common/Table';
import GradientRuler from '../common/GradientRuler';
import {measureGetter, numDomainGetter, xyDomainGetter} from '../../utils/WidgetHelpers';
import {getColType, isTextCol} from '../../utils/ColTypes';
import {formatterGetter} from '../../utils/Formatters';
import {UNSELECTED_COLOR, genColorGetter} from '../../utils/Colors';
import {default_duration, animate, getBubbleTransistData} from '../../utils/Animate';
import {DEFAULT_WIDGET_WRAPPER_WIDTH, DEFAULT_WIDGET_WRAPPER_HEIGHT} from '../../utils/Layout';
import {BubbleChartProps} from './types';
import {checkIsSelected} from './filter';
import './style.scss';

// BubbleCHART realted consts
const margin = {top: 40, right: 20, bottom: 30, left: 80};
const xLabelHeight = 20;
const xLabelMarginTop = 20;
const defaultFontSize = 10;
const tooltipTitleFontSize = 10;
const toolTipContentMarginRight = 5;
const tooltipDimensionMaxWidth = 150;
const tooltipTitleTextMaxWidth = 80;
const toolTipMeasureMinWidth = 80;

const BubbleChart = (props: BubbleChartProps) => {
  const theme = useTheme();
  const {showTooltip, hideTooltip} = useContext(rootContext);
  const {nls} = useContext(I18nContext);
  const {
    config,
    wrapperWidth = DEFAULT_WIDGET_WRAPPER_WIDTH,
    wrapperHeight = DEFAULT_WIDGET_WRAPPER_HEIGHT,
    data = [],
    onClick = () => {},
    dataMeta,
  } = props;
  const {colorKey = '', dimensions = [], measures = [], ruler = {}, rulerBase, filter} = config;
  const allSvgHeight = wrapperHeight - xLabelHeight - xLabelMarginTop;
  const svgWidth = wrapperWidth;
  const svgHeight = allSvgHeight;
  const width = svgWidth - margin.left - margin.right;
  const height = svgHeight - margin.top - margin.bottom;
  const xMeasure = measureGetter(config, 'x')!;
  const xMeasureFormatter = formatterGetter(xMeasure);
  const yMeasure = measureGetter(config, 'y')!;
  const yMeasureFormatter = formatterGetter(yMeasure)!;
  const colorMeasure = measureGetter(config, 'color')! || {};
  const sizeMeasure = measureGetter(config, 'size')! || {};
  const defaultRadius = Math.max((12 * width * height) / (800 * 800), 12);
  const radiusRange = [defaultRadius, defaultRadius * 3];
  const xAxisLabel = xMeasure.label;
  const yAxisLabel = yMeasure.isRecords
    ? nls.label_widgetEditor_recordOpt_label_measure
    : yMeasure.label;
  // xAxis | yAxis | SizeScale | Color
  const [x, y, size] = [
    xyDomainGetter(data, xMeasure.as, [0, width]),
    xyDomainGetter(data, yMeasure.as, [height, 0]),
    numDomainGetter(data, sizeMeasure.as, radiusRange),
  ];
  const xAxis: any = axisBottom(x);
  const yAxis: any = axisLeft(y);
  xAxis.tickFormat(xMeasureFormatter);
  yAxis.tickFormat(yMeasureFormatter);
  const getColor = genColorGetter(config);
  const [preRenderData, setPreRenderData] = useState<any>([]);
  // size Range getter;
  const textPositionGetter = (
    text: string,
    fontSize: number,
    xPos: number,
    yPos: number,
    radius: number
  ) => {
    const len = text.length;
    const textXPos = xPos - (fontSize * len) / 4;
    const textYPos = yPos + radius / 2 || 0;

    return {textXPos, textYPos};
  };
  const renderData =
    data.length > 0
      ? data.map((item: any) => {
          const colorValue = getColor(item[colorMeasure.as], theme.palette.background.default);
          const xPos = x(item[xMeasure.as]);
          const yPos = y(item[yMeasure.as]);
          const isSelected = checkIsSelected(item, config);
          const color = isSelected ? colorValue : UNSELECTED_COLOR;
          const radius = sizeMeasure.as
            ? size(Number.parseFloat(item[sizeMeasure.as])) || defaultRadius
            : defaultRadius;
          const text = dimensions
            .map((dimension: any) => {
              const {maxbins, extent = [], type, extract} = dimension;
              const [min, max] = extent;
              const colType = getColType(type);
              const formatter = formatterGetter(dimension);
              let rangeMin, rangeMax;
              switch (colType) {
                case 'date':
                  if (!extract) {
                    rangeMin = new Date(item[dimension.as]);
                  }
                  if (extract) {
                    rangeMin = item[dimension.as];
                  }
                  break;
                case 'number':
                  const gap = (max - min) / maxbins;
                  const grpNum = item[dimension.as];
                  rangeMin = min + gap * grpNum;
                  rangeMax = rangeMin + gap;
                  break;
                case 'text':
                  rangeMin = item[dimension.as];
                  break;
                default:
                  break;
              }
              if (
                rangeMin === undefined ||
                Number.isNaN(rangeMin) ||
                rangeMin.toString() === 'Invalid Date'
              ) {
                return '...';
              }
              return `${formatter(rangeMin)}${rangeMax ? ` ~ ${formatter(rangeMax)}` : ''}`;
            })
            .join(' / ');
          const {textXPos, textYPos} = textPositionGetter(
            text,
            defaultFontSize,
            xPos,
            yPos,
            radius
          );
          return {
            xPos,
            yPos,
            radius,
            color,
            text,
            textXPos,
            textYPos,
            originData: item,
          };
        })
      : [];

  const effectFactors = JSON.stringify([
    dataMeta && dataMeta.loading,
    wrapperWidth,
    wrapperHeight,
    colorKey,
    rulerBase,
    ruler,
    filter,
    dimensions,
  ]);

  useEffect(() => {
    if (preRenderData.length === 0) {
      setPreRenderData(renderData.length > 0 ? renderData : preRenderData);
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
          getBubbleTransistData(preRenderData, renderData, elapsed, default_duration)
        );
      },
    });
    ts.start();
    return () => {
      ts.stop();
    };
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectFactors]);

  const titleGetter = (tooltipData: any) => {
    const {dimensionTitle, measureTitles} = tooltipData;
    return (
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          fontSize: `${tooltipTitleFontSize}px`,
        }}
      >
        <h3
          style={{
            width: `${tooltipDimensionMaxWidth}px`,
            marginRight: `${toolTipContentMarginRight}px`,
            overflow: 'hidden',
            whiteSpace: 'nowrap',
            textOverflow: 'ellipsis',
          }}
        >
          {dimensionTitle}
        </h3>
        {measureTitles.map((content: any, index: number) => {
          return (
            <div
              key={index}
              style={{
                width: `${toolTipMeasureMinWidth}px`,
                marginRight: `${toolTipContentMarginRight}px`,
              }}
            >
              <h3
                style={{
                  maxWidth: `${tooltipTitleTextMaxWidth}px`,
                  overflow: 'hidden',
                  whiteSpace: 'nowrap',
                  textOverflow: 'ellipsis',
                }}
              >
                {content}
              </h3>
            </div>
          );
        })}
      </div>
    );
  };

  const contentGetter = (tooltipData: any) => {
    const {dimensionTitle, measureTitles, dimensionValueGrp, measureValueGrp} = tooltipData;
    const mTitles = measureTitles.map((item: any) => {
      const {label, type, as, expression, isCustom, isRecords} = item;
      const isText = isTextCol(type);
      const isNotShowAgg = isText || isCustom;
      return {
        field: as,
        name: isNotShowAgg
          ? `${label}(${nls.label_BubbleChart_count})`
          : `${isRecords ? nls.label_widgetEditor_recordOpt_label_measure : label}(${
              nls[`label_widgetEditor_expression_${expression}`]
            })`,
        format: null,
      };
    });
    const def = [
      {
        field: dimensionTitle,
        name: dimensionTitle,
        format: null,
      },
      ...mTitles,
    ];
    const data = dimensionValueGrp.map((dValue: any, index: number) => {
      const mValues = measureValueGrp[index];
      const mData: any = {};
      mValues.forEach((v: any, _index: number) => {
        mData[measureTitles[_index].as] = v;
      });
      return {
        [dimensionTitle]: dValue,
        ...mData,
      };
    });
    return (
      <div
        style={{
          maxHeight: '150px',
          overflow: 'auto',
        }}
      >
        <Table def={def} data={data} length={data.length} />
      </div>
    );
  };

  const onCircleMouseOver = (item: any, e: MouseEvent) => {
    const items = getDisplayTooltipItems(defaultRadius, renderData, item);
    const tootipDatas = getTooltipDatasByItems(items);
    showTooltip({
      position: {event: e},
      tooltipData: tootipDatas,
      titleGetter,
      contentGetter,
      isShowTitle: false,
    });
  };

  const getDisplayTooltipItems = (radius: number, datas: any[], hoverItem: any) => {
    const items = datas.filter((target: any) => {
      let xDistant = Math.abs(target.xPos - hoverItem.xPos),
        yDistant = Math.abs(target.yPos - hoverItem.yPos),
        lineDistance = Math.pow(xDistant, 2) + Math.pow(yDistant, 2),
        radiusPow = Math.pow(radius, 2),
        isShow = radiusPow > lineDistance;
      return isShow;
    });
    return items;
  };

  const getTooltipDatasByItems = (items: any[]) => {
    const dimensionTitle = dimensions.map((dimension: any) => dimension.value).join(' / ');
    const measureTitles = measures.map((measure: any) => {
      const {type, label, as, expression, isCustom, isRecords} = measure;
      return {
        type,
        label,
        as,
        expression,
        isCustom,
        isRecords,
      };
    });
    const dimensionValueGrp = items.map((item: any) => {
      return (
        <div
          style={{
            display: 'flex',
            justifyContent: 'start',
            alignItems: 'center',
          }}
        >
          <div
            style={{
              backgroundColor: item.color,
              width: '10px',
              height: '10px',
              marginRight: '5px',
            }}
          ></div>
          <div>{item.text}</div>
        </div>
      );
    });
    const measureValueGrp = items.map((item: any) => {
      const measureValues = measures.map((measure: any) => {
        const formatter = formatterGetter(measure);
        return formatter(item.originData[measure.as]);
      });
      return measureValues;
    });
    return {
      dimensionTitle,
      measureTitles,
      dimensionValueGrp,
      measureValueGrp,
    };
  };

  // render
  return (
    <div
      className={`z-chart z-bubble-chart`}
      style={{
        height: wrapperHeight,
        width: wrapperWidth,
        backgroundColor:theme.palette.background.paper,
        overflow: 'hidden',
        fontSize: `${defaultFontSize}px`,
        position: 'relative',
      }}
    >
      {preRenderData.length > 0 && (
        <svg width={wrapperWidth} height={wrapperHeight}>
          {/* xAxis, yAxis */}
          <g width={width} height={height} transform={`translate(${margin.left}, ${margin.top})`}>
            <g className="axis axis--x" pointerEvents="none">
              <g
                className="grid-line"
                transform={`translate(0, ${height})`}
                ref={node => select(node).call(xAxis)}
              />
              <g
                className="grid-line"
                transform={`translate(0, ${height})`}
                ref={node =>
                  select(node).call(
                    xAxis
                      .tickSizeInner(-height)
                      .tickSizeOuter(0)
                      .tickFormat('')
                  )
                }
              />
              <text
                fill="#000"
                y={svgHeight - 30}
                x={wrapperWidth / 2}
                dy="0.71em"
                textAnchor="end"
              >
                {xAxisLabel}
              </text>
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
                fill="#000"
                transform="rotate(-90)"
                y={-margin.left + 10}
                x={-50}
                dy="0.71em"
                textAnchor="end"
              >
                {yAxisLabel}
              </text>
            </g>
            {/* Bubbles and Texts */}
            {(preRenderData.length > 0 ? preRenderData : renderData).map(
              (item: any, index: number) => {
                const {xPos, yPos, radius, color, text, textXPos, textYPos} = item;
                return (
                  <Fragment key={index}>
                    <circle
                      cy={yPos}
                      cx={xPos}
                      r={radius}
                      stroke={color}
                      fill={color}
                      onMouseOver={(e: any) => {
                        onCircleMouseOver(item, e);
                      }}
                      onMouseLeave={hideTooltip}
                      onClick={(e: any) => {
                        onClick(item.originData);
                      }}
                    />
                    <text
                      y={textYPos}
                      x={textXPos}
                      style={{
                        fill: '#333',
                        // strokeWidth: '2px',
                        // stroke: 'black',
                        // paintOrder: 'stroke',
                        // strokeOpacity: 0.3,
                      }}
                    >
                      {text}
                    </text>
                  </Fragment>
                );
              }
            )}
          </g>
        </svg>
      )}
      <GradientRuler {...props} />
    </div>
  );
};

export default BubbleChart;
