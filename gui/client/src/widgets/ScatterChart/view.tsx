import React, {FC, useEffect, useRef, useContext} from 'react';
import {useTheme} from '@material-ui/core/styles';
import {scaleLinear} from 'd3';
import {queryContext} from '../../contexts/QueryContext';
import {rootContext} from '../../contexts/RootContext';
import {ScatterChartConfig} from './types';
import {Measure} from '../../types';
import {CONFIG} from '../../utils/Consts';
import {throttle, measureGetter} from '../../utils/WidgetHelpers';
import {rectHandler, getMarkerPos} from './filter';
import ScatterChart from '../ScatterChart';

const DEFAULT_RADIUS = 4;
const ScatterChartView: FC<ScatterChartConfig> = props => {
  const theme = useTheme();
  const {getRowBySql} = useContext(queryContext);
  const {config, setConfig, dataMeta} = props;
  const {showTooltip, hideTooltip} = useContext(rootContext);
  const {width, height} = config;
  const xMeasure = measureGetter(config, 'x')!;
  const yMeasure = measureGetter(config, 'y')!;
  const xDomain = xMeasure.domain!;
  const yDomain = yMeasure.domain!;
  const xStaticDomain = xMeasure.staticDomain!;
  const yStaticDomain = yMeasure.staticDomain!;
  const [x, y] = [
    scaleLinear()
      .range([0, width])
      .domain(xDomain!),
    scaleLinear()
      .range([height, 0])
      .domain(yDomain!),
  ];

  const useTooltip = useRef<any>(true);
  const marker = useRef<SVGCircleElement>(null);
  const markerData = useRef<any>(null);

  const _onRectChange = (width: number, height: number) => {
    setConfig({type: CONFIG.UPDATE, payload: rectHandler({config, width, height})});
  };

  const _setConfig = throttle(setConfig, 500);
  const onZoomEnd = ({newScaleX, newScaleY}: any) => {
    _setConfig({
      type: CONFIG.UPDATE_AXIS_RANGE,
      payload: {id: config.id, x: newScaleX.domain(), y: newScaleY.domain()},
    });
  };
  const _reset = () => {
    setConfig({
      type: CONFIG.UPDATE_AXIS_RANGE,
      payload: {id: config.id, x: xMeasure.staticDomain, y: yMeasure.staticDomain},
    });
  };

  const contentGetter = (tooltipData: any) => {
    const cdds = config.measures.map((m: any) => m.value).concat(config.popupItems);
    const keys = Object.keys(tooltipData).filter(
      (k: string) => !!cdds.find((cdd: string) => cdd === k)
    );
    return (
      <>
        <ul style={{minWidth: '300px'}}>
          {keys.map((k: string) => (
            <li style={{textAlign: 'left', display: 'flex'}} key={k}>
              <span style={{textAlign: 'right', width: '50%', marginRight: '20px'}}>
                <strong>{`${k}:`}</strong>
              </span>
              <span>{`${tooltipData[k]}`}</span>
            </li>
          ))}
        </ul>
      </>
    );
  };
  const _getPoint = ({event, xLen, yLen}: any) => {
    const sql = _sqlGetter({config, xLen, yLen, dataMeta});
    useTooltip.current &&
      getRowBySql(sql).then((rows: any) => {
        if (rows && rows[0]) {
          // showTooltip
          const tooltipData = rows[0];
          showTooltip({
            position: {event},
            tooltipData,
            contentGetter,
            isShowTitle: false,
          });
          markerData.current = tooltipData;

          _setMarker({data: tooltipData, config, x, y});
        }
      });
  };
  const onZooming = () => {
    hideTooltip(undefined, 0);
    _setMarker({});
  };
  const _setMarker = ({data, config, x, y}: any) => {
    if (data) {
      const posRes = getMarkerPos({data, config, x, y});
      marker.current && marker.current.setAttribute('cx', `${posRes.x}px`);
      marker.current && marker.current.setAttribute('cy', `${posRes.y}px`);
      marker.current &&
        marker.current.setAttribute('fill', posRes.color || theme.palette.background.default);
    } else {
      marker.current && marker.current.setAttribute('cx', `${-99999}px`);
      marker.current && marker.current.setAttribute('cy', `${-99999}px`);
    }
  };
  const _sqlGetter = ({config, xLen, yLen, dataMeta}: any) => {
    const circleR = _getValidCircle();
    const exprs = config.measures.map((m: Measure) => `${m.value} as ${m.as}`).join(', ');
    const where = dataMeta.sql.split('WHERE')[1];
    const globalFilter = where ? where.split('LIMIT')[0] : '';
    const filter = _singlePointFilterGetter({circleR, x: xLen, y: yLen});
    return `SELECT *, ${exprs} FROM ${config.source} WHERE (${filter}${
      globalFilter ? `AND ${globalFilter}` : ''
    }) LIMIT 1`;
  };
  const _singlePointFilterGetter = ({circleR, x, y}: any) => {
    return `(${xMeasure.value} BETWEEN ${x - circleR.x} AND ${x + circleR.x}) AND (${
      yMeasure.value
    } BETWEEN ${y - circleR.y} AND ${y + circleR.y})`;
  };
  const _getValidCircle = () => {
    const x = scaleLinear()
      .range([0, width])
      .domain([0, xDomain[1] - xDomain[0]]);
    const y = scaleLinear()
      .range([0, height])
      .domain([0, yDomain[1] - yDomain[0]]);
    return {x: x.invert(DEFAULT_RADIUS), y: y.invert(DEFAULT_RADIUS)};
  };
  const onMouseMove = throttle(_getPoint, 700);
  const onMouseLeave = () => {
    onMouseMove.cancel();
    hideTooltip();
    if (marker.current) {
      marker.current.setAttribute('cx', `${-99999999999999}px`);
      marker.current.setAttribute('cy', `${-99999999999999}px`);
    }
  };
  const isFirstRun = useRef(true);
  useEffect(() => {
    if (isFirstRun.current) {
      isFirstRun.current = false;
      return;
    }
    const isStaticRangeChange =
      JSON.stringify(xDomain) !== JSON.stringify(xStaticDomain) ||
      JSON.stringify(yDomain) !== JSON.stringify(yStaticDomain);

    if (isStaticRangeChange) {
      setConfig({
        type: CONFIG.UPDATE_AXIS_RANGE,
        payload: {id: config.id, x: xStaticDomain, y: yStaticDomain},
      });
    }
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify([xStaticDomain, yStaticDomain])]);
  return (
    <ScatterChart
      config={config}
      setConfig={setConfig}
      mode={props.mode}
      setMode={props.setMode}
      data={props.data}
      dataMeta={props.dataMeta}
      dashboard={props.dashboard}
      wrapperWidth={props.wrapperWidth}
      wrapperHeight={props.wrapperHeight}
      onZooming={onZooming}
      onZoomEnd={onZoomEnd}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
      onRectChange={_onRectChange}
      reset={_reset}
      radius={DEFAULT_RADIUS}
      ref={marker}
    />
  );
};

export default ScatterChartView;
