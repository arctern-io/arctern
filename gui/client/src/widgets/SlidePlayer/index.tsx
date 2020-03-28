import React, {FC, useRef, useEffect, useState, useMemo} from 'react';
import {useTheme} from '@material-ui/core/styles';
import Brush from '../common/Brush';
import xAxis from '../Utils/XAxis';
import {dimensionGetter, brushDataGetter, dimensionTypeGetter} from '../../utils/WidgetHelpers';
import {nextTimeWrapper, nextTextOrNumberWrapper} from './Helper';
import PlayCircleOutlineIcon from '@material-ui/icons/PlayCircleOutline';
import PauseCircleOutlineIcon from '@material-ui/icons/PauseCircleOutline';
import './style.scss';
import {COLUMN_TYPE} from '../../utils/Consts';
import {SlidePlayerProps} from './types';

const margin = {top: 30, right: 20, bottom: 20, left: 20};
const PLAY_ICON_WIDTH = 50;
const Timeline_PLAY_TIMEOUT = 1500;

const SlidePlayer: FC<SlidePlayerProps> = props => {
  const theme = useTheme();
  const xAxisContainer = useRef<SVGGElement>(null);
  const playingTimeout = useRef<any>(null);
  const [position, setPosition] = useState<any[]>([]);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const {wrapperWidth = 0, wrapperHeight = 0, onRangeChange, data, config, dataMeta} = props;

  const svgHeight = 70;
  const svgWidth = wrapperWidth - PLAY_ICON_WIDTH;
  const height = svgHeight - margin.top - margin.bottom;
  const width = svgWidth - margin.left - margin.right;

  const xDimension = dimensionGetter(config, 'x')!;
  const typeX = dimensionTypeGetter(xDimension);
  const isTimeChart = typeX === COLUMN_TYPE.DATE;
  const {binningResolution} = xDimension;
  const {brush} = brushDataGetter(xDimension, config.filter, false);

  const {xAxisInstance, x} = useMemo(() => {
    const domain = data[0].options;
    const newXAxis = new xAxis(xDimension, width);
    const formatter = newXAxis.defaultFormatter;
    newXAxis.setDomain(domain);
    newXAxis.setTicks(1000, xDimension, (data: any, index: number, allTicks: []) => {
      return index === 0 ||
        index === allTicks.length - 1 ||
        index % Math.floor(allTicks.length / 4) === 0
        ? formatter(data)
        : '';
    });
    return {
      xAxisInstance: newXAxis,
      x: newXAxis.xScale,
    };
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(xDimension), width]);

  // stop playing when clean filter
  useEffect(() => {
    if (Object.keys(config.filter).length === 0) {
      setIsPlaying(false);
    }
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(config.filter)]);

  const effectFactors = [
    dataMeta && dataMeta.loading,
    wrapperHeight,
    wrapperWidth,
    svgHeight,
    JSON.stringify(xDimension),
  ];

  // update widget
  useEffect(() => {
    if (!dataMeta || dataMeta.loading || !xAxisInstance) {
      return;
    }
    handleBrushMove(brush);

    xAxisInstance.update(xAxisContainer.current as SVGGElement);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(effectFactors)]);

  // start brush playing
  useEffect(() => {
    if (!isPlaying) {
      clearTimeout(playingTimeout.current);
      return;
    }
    const nextTextGetter = nextTextOrNumberWrapper(false);
    const nextNumberGetter = nextTextOrNumberWrapper(false);
    const nextTimeGetter = nextTimeWrapper(false);

    playingTimeout.current = setTimeout(() => {
      let nextBrush = [];
      switch (typeX) {
        case COLUMN_TYPE.DATE:
          nextBrush = nextTimeGetter(brush, binningResolution!, x.domain());
          break;
        case COLUMN_TYPE.TEXT:
          nextBrush = nextTextGetter(width, x, brush, x.step());
          break;
        case COLUMN_TYPE.NUMBER:
        default:
          const {maxbins, max, min, extract} = xDimension;
          const numStep =
            x(
              extract
                ? (min! as number) + 1
                : (min! as number) + ((max! as number) - (min! as number)) / maxbins!
            ) - x(min);
          nextBrush = nextNumberGetter(width, x, brush, numStep);
          break;
      }
      onRangeChange(config, nextBrush);
    }, Timeline_PLAY_TIMEOUT);

    return () => {
      clearTimeout(playingTimeout.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, JSON.stringify(position)]);

  const handleBrushUpdate = (range: [string | number | Date | boolean]) => {
    onRangeChange(config, isTimeChart ? range.map((v: any) => Date.parse(v)) : range);
    setIsPlaying(false);
  };
  const handleBrushMove = (range: any[]) => {
    const formatter = xAxisInstance.defaultFormatter;
    setPosition(range.map((v: any) => formatter(typeX === COLUMN_TYPE.NUMBER ? Math.round(v) : v)));
  };
  const togglePlaying = () => {
    setIsPlaying(pre => !pre);
  };

  return (
    <div
      style={{width: wrapperWidth, height: svgHeight, background: theme.palette.background.paper}}
      className="slide-player"
    >
      <div className="status-wrapper">
        <span style={{marginRight: '10px'}}>{xDimension.label}</span>
        {position.join(' - ')}
      </div>
      <div className="content">
        <div className="control-wrapper">
          {!isPlaying ? (
            <PlayCircleOutlineIcon
              fontSize="large"
              className="control-icon"
              color="primary"
              onClick={togglePlaying}
            ></PlayCircleOutlineIcon>
          ) : (
            <PauseCircleOutlineIcon
              onClick={togglePlaying}
              fontSize="large"
              color="primary"
              className="control-icon"
            ></PauseCircleOutlineIcon>
          )}
        </div>
        <svg className="slider-container" width={svgWidth} height={svgHeight}>
          <g
            width={svgWidth - margin.left * 2}
            height={height}
            transform={`translate(${margin.left}, ${margin.top})`}
          >
            <g className="axis axis--x" pointerEvents="none">
              <g className="grid-line" transform={`translate(0,${height})`}></g>
              <g ref={xAxisContainer} transform={`translate(0, ${height})`} />
            </g>
          </g>
          <g>
            <rect
              width={svgWidth - margin.left * 2}
              height={height}
              style={{fill: 'transparent', strokeWidth: 1, stroke: '#2196f3'}}
              transform={`translate(${margin.left}, ${margin.top})`}
            ></rect>
          </g>
        </svg>
        <svg
          width={svgWidth}
          height={svgHeight}
          style={{
            position: 'absolute',
            left: `${PLAY_ICON_WIDTH}px`,
            top: '0px',
          }}
        >
          <g transform={`translate(${margin.left}, ${margin.top})`}>
            {x && (
              <Brush
                x={x}
                dataMeta={dataMeta}
                config={config}
                isTimeChart={isTimeChart}
                xDimension={xDimension}
                width={width}
                height={height}
                filter={config.filter}
                onBrushMove={handleBrushMove}
                onBrushUpdate={handleBrushUpdate}
              />
            )}
          </g>
        </svg>
      </div>
    </div>
  );
};

export default SlidePlayer;
