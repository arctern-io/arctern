import React, {useContext, useRef, useEffect} from 'react';
import {ScatterChartProps} from './types';
import {
  select,
  axisBottom,
  axisLeft,
  zoom as d3Zoom,
  event,
  scaleLinear,
  mouse,
  zoomIdentity,
} from 'd3';
import {useTheme} from '@material-ui/core/styles';
import {I18nContext} from '../../contexts/I18nContext';
import {measureGetter} from '../../utils/WidgetHelpers';
import {formatterGetter} from '../../utils/Formatters';
import {DEFAULT_WIDGET_WRAPPER_WIDTH, DEFAULT_WIDGET_WRAPPER_HEIGHT} from '../../utils/Layout';
import GradientRuler from '../common/GradientRuler';
import './style.scss';

// ScatterChart realted consts
export const margin = {top: 40, right: 20, bottom: 30, left: 80};
const xLabelHeight = 20;
const xLabelMarginTop = 20;
const defaultFontSize = 10;

const ScatterChart = React.forwardRef(
  (props: ScatterChartProps, ref: React.Ref<SVGCircleElement>) => {
    const {nls} = useContext(I18nContext);
    const {
      config,
      wrapperWidth = DEFAULT_WIDGET_WRAPPER_WIDTH,
      wrapperHeight = DEFAULT_WIDGET_WRAPPER_HEIGHT,
      data,
      dataMeta,
      onZooming = () => {},
      onZoomEnd = () => {},
      onMouseMove = () => {},
      onMouseLeave = () => {},
      onRectChange = () => {},
      reset = () => {},
      radius = 0,
    } = props;
    const theme = useTheme();
    const {width, height, layout} = config;
    const allSvgHeight = wrapperHeight - xLabelHeight - xLabelMarginTop;
    const svgWidth = wrapperWidth;
    const svgHeight = allSvgHeight;
    const xMeasure = measureGetter(config, 'x')!;
    const yMeasure = measureGetter(config, 'y')!;

    const xDomain = xMeasure.domain;
    const yDomain = yMeasure.domain;
    const domains = [xMeasure.domain, yMeasure.domain];
    const xStaticDomain = xMeasure.staticDomain;
    const yStaticDomain = yMeasure.staticDomain;
    const staticDomains = [xStaticDomain, yStaticDomain];

    const xMeasureFormatter = formatterGetter(xMeasure);
    const yMeasureFormatter = formatterGetter(yMeasure);
    const xAxisLabel = xMeasure.label;
    const yAxisLabel = yMeasure.isRecords
      ? nls.label_widgetEditor_recordOpt_label_measure
      : yMeasure.label;

    const [x, y] = [
      scaleLinear()
        .range([0, width])
        .domain(xDomain!),
      scaleLinear()
        .range([height, 0])
        .domain(yDomain!),
    ];

    const svgRoot = useRef<any>(null);
    const xAxisContainer = useRef<any>(null);
    const yAxisContainer = useRef<any>(null);
    const zoomArea = useRef<any>(null);

    const xGridLine = useRef<any>(null);
    const yGridLine = useRef<any>(null);
    const canvasContainer = useRef<any>(null);
    const transformPos = useRef<any>({left: 0, top: 0});
    const canvas = useRef<any>(null);
    const imageObj = useRef<any>(null);
    const lastTransform = useRef<any>({k: 1});
    const useTooltip = useRef<any>(true);

    const xAxis: any = axisBottom(x);
    const yAxis: any = axisLeft(y);

    function bindNewZoom({x, y}: any) {
      x.domain(xStaticDomain);
      y.domain(yStaticDomain);
      transformPos.current = {left: 0, top: 0};
      const zoom = d3Zoom()
        .on('zoom', zooming)
        .on('end', () => {
          useTooltip.current = true;
          const newScaleX = event.transform.rescaleX(x);
          const newScaleY = event.transform.rescaleY(y);
          transformPos.current = {left: event.transform.x, top: event.transform.y};
          onZoomEnd({newScaleX, newScaleY});
        });
      select(zoomArea.current)
        .call(zoom.transform, zoomIdentity)
        .call(zoom);
    }
    function updateZoom({x, y}: any) {
      x.domain(xDomain);
      y.domain(yDomain);
      transformPos.current = {left: 0, top: 0};
      const zoom = d3Zoom()
        .on('zoom', zooming)
        .on('end', () => {
          useTooltip.current = true;
          const newScaleX = event.transform.rescaleX(x);
          const newScaleY = event.transform.rescaleY(y);
          transformPos.current = {left: event.transform.x, top: event.transform.y};
          onZoomEnd({newScaleX, newScaleY});
        });
      select(zoomArea.current).call(zoom);
    }
    const _setNewAxises = ({new_xScale, new_yScale}: any) => {
      select(xAxisContainer.current).call(xAxis.scale(new_xScale).tickFormat(xMeasureFormatter));
      select(yAxisContainer.current).call(yAxis.scale(new_yScale).tickFormat(yMeasureFormatter));
      select(xGridLine.current).call(
        xAxis
          .scale(new_xScale)
          .tickSizeInner(-height)
          .tickSizeOuter(0)
          .tickFormat('')
      );
      select(yGridLine.current).call(
        yAxis
          .scale(new_yScale)
          .tickFormat('')
          .tickSizeInner(-width)
          .tickSizeOuter(0)
      );
    };
    const _setImagePos = (transform: any) => {
      const {x, y, k} = transform;
      const translationChart = k === lastTransform.current.k;
      if (translationChart) {
        canvasContainer.current.style.left = `${x - transformPos.current.left}px`;
        canvasContainer.current.style.top = `${y - transformPos.current.top}px`;
      }
      lastTransform.current = transform;
    };

    const _initImagPos = () => {
      canvasContainer.current.style.left = `0px`;
      canvasContainer.current.style.top = `0px`;
    };

    const zooming = () => {
      onZooming();
      useTooltip.current = false;
      const new_xScale = event.transform.rescaleX(x);
      const new_yScale = event.transform.rescaleY(y);
      _setNewAxises({new_xScale, new_yScale});
      _setImagePos(event.transform);
    };

    const _onMouseMove = function() {
      let [xPos, yPos] = mouse(event.target);
      const isValidX = xPos >= 0 && xPos <= width;
      const isValidY = yPos >= 0 && yPos <= height;
      if (isValidX && isValidY) {
        onMouseMove({event, config, xLen: x.invert(xPos), yLen: y.invert(yPos)});
      }
    };

    const _reset = () => {
      bindNewZoom({x, y});
      reset();
    };

    const rangeEffectors = [width, height, layout];
    useEffect(() => {
      x.range([0, width]);
      y.range([height, 0]);
      updateZoom({x, y});
      //eslint-disable-next-line react-hooks/exhaustive-deps
    }, [JSON.stringify(rangeEffectors)]);

    // Add EventListener(mousemove, mouseleave) on SVG_Node
    const pointEffectors = [width, height, domains, dataMeta];
    useEffect(() => {
      _setNewAxises({new_xScale: x, new_yScale: y});
      let _zoomArea = select(zoomArea.current);
      _zoomArea.on('mousemove', _onMouseMove);
      _zoomArea.on('mouseleave', () => onMouseLeave());
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [JSON.stringify([pointEffectors])]);

    // Bind Zoom Event and DO NOT change while zooming
    const isFirstRun = useRef(true);
    useEffect(() => {
      if (isFirstRun.current) {
        isFirstRun.current = false;
        return;
      }
      bindNewZoom({x, y});
      //eslint-disable-next-line react-hooks/exhaustive-deps
    }, [JSON.stringify([staticDomains])]);

    // set config.width and config.height when wrapperRect change
    useEffect(() => {
      const newWidth = svgWidth - margin.left - margin.right;
      const newHeight = svgHeight - margin.top - margin.bottom;
      if (config.width !== Math.floor(newWidth) || config.height !== Math.floor(newHeight)) {
        onRectChange(newWidth, newHeight);
      }
      //eslint-disable-next-line react-hooks/exhaustive-deps
    }, [wrapperWidth, wrapperHeight]);

    // Set Image when Image update
    useEffect(() => {
      // Create and set Image
      imageObj.current = new Image();
      imageObj.current.onload = function() {
        const context = canvas.current.getContext('2d');
        _initImagPos();
        context.clearRect(0, 0, width, height);
        context.drawImage(imageObj.current, 0, 0);
      };
      imageObj.current.src = data;
      //eslint-disable-next-line react-hooks/exhaustive-deps
    }, [data]);

    return (
      <div
        className={`z-chart z-scatter-chart`}
        style={{
          height: wrapperHeight,
          width: wrapperWidth,
          fontSize: `${defaultFontSize}px`,
          background: theme.palette.background.paper,
        }}
      >
        <button onClick={() => _reset()}>{nls.label_ScatterChart_rest}</button>
        <div
          className="svg-container"
          style={{
            width: svgWidth,
            height: svgHeight,
          }}
        >
          <svg width={svgWidth} height={svgHeight}>
            <circle
              ref={ref}
              r={radius}
              cx={-99999999}
              cy={-99999999}
              fill={'red'}
              stroke="white"
            />
          </svg>
        </div>
        <svg width={wrapperWidth} height={wrapperHeight} ref={svgRoot}>
          {/* xAxis, yAxis */}
          <g width={width} height={height} transform={`translate(${margin.left}, ${margin.top})`}>
            <g className="axis axis--x" pointerEvents="none">
              <g ref={xGridLine} className="grid-line" transform={`translate(0, ${height})`} />
              <g ref={xAxisContainer} transform={`translate(0, ${height})`} />
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
              <g ref={yGridLine} className="grid-line" />
              <g ref={yAxisContainer} />
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
            <rect width={width} height={height} ref={zoomArea} fill="transparent" />
          </g>
        </svg>
        <div
          className="canvas-cover"
          style={{
            width,
            height,
            top: margin.top,
            left: margin.left,
          }}
        >
          <div
            className="canvas-container"
            ref={canvasContainer}
            style={{
              width,
              height,
            }}
          >
            <canvas ref={canvas} width={width} height={height}></canvas>
          </div>
        </div>
        <GradientRuler {...props} />
      </div>
    );
  }
);

export default ScatterChart;
