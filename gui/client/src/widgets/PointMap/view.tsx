import React, {FC, useRef, useState, useContext} from 'react';
import {useTheme} from '@material-ui/core/styles';
import {WidgetConfig, ColorItem} from '../../types';
import distance from '@turf/distance';
import Paper from '@material-ui/core/Paper';
import mapboxgl from 'mapbox-gl';
import {PointMapProps} from './types';
import Spinner from '../../components/common/Spinner';
import MapboxGl from '../common/MapboxGl';
import {MapChartConfig} from '../common/MapChart.type';
import {queryContext} from '../../contexts/QueryContext';
import {I18nContext} from '../../contexts/I18nContext';
import {
  measureGetter,
  popupContentGetter,
  popupContentBuilder,
  throttle,
  parseExpression,
} from '../../utils/WidgetHelpers';
import {CONFIG} from '../../utils/Consts';
import {cloneObj} from '../../utils/Helpers';
import {delayRunFunc} from '../../utils/EditorHelper';
import {markerPosGetter, KEY} from '../Utils/Map';
import {mapUpdateConfigHandler, drawUpdateConfigHandler} from '../Utils/filters/map';
import {DEFAULT_COLOR, genColorGetter, isGradientType} from '../../utils/Colors';
const PointMapNormal: FC<PointMapProps> = props => {
  const theme = useTheme();
  const {getRowBySql} = useContext(queryContext);
  const {nls} = useContext(I18nContext);
  const {config, setConfig, dataMeta} = props;
  const [isLoading, setLoading] = useState<boolean>(false);

  const paperRef = useRef<any>();
  const detailRef = useRef<any>();
  const popup = useRef<any>(null);
  const marker = useRef<any>(null);
  const pointRequest = useRef<any>(null);
  // get draws
  const copiedConfig = cloneObj(config);
  const {popupItems = [], measures = []} = copiedConfig;
  const [allowPopUp, setAllowPopUp] = useState(popupItems.length + measures.length > 0);

  // map update on bounds change
  const onMapUpdate = (map: any, container: any) => {
    const center = map.getCenter();
    const bounds = map.getBounds();
    const zoom = map.getZoom();
    const boundingClientRect = container.getBoundingClientRect();
    setConfig({
      type: CONFIG.UPDATE,
      payload: mapUpdateConfigHandler(config, {
        boundingClientRect,
        zoom,
        center,
        bounds,
      }),
    });
  };

  // on draw update
  const onDrawUpdate = throttle((draws: any) => {
    setConfig({
      type: CONFIG.UPDATE,
      payload: drawUpdateConfigHandler(config, draws),
    });
  }, 50);

  const _geoJsonPointGetter = (point: any): any => {
    const {lng, lat} = point;
    return {
      type: 'Feature',
      properties: {},
      geometry: {
        type: 'Point',
        coordinates: [lng, lat],
      },
    };
  };
  const _disPerPixelGetter = (map: any, container: HTMLElement, distanceGetter: Function) => {
    const {width, height} = container.getBoundingClientRect();
    const mapPixcelCornerDis = Math.sqrt(width * width + height * height);
    const {_ne, _sw} = map.getBounds();
    const mapConerDis = distanceGetter(_geoJsonPointGetter(_ne), _geoJsonPointGetter(_sw));
    const disPerPixel = mapConerDis / mapPixcelCornerDis;
    return disPerPixel;
  };
  const _genPointMapPointSql = (
    center: any,
    pointCircleR: number,
    dataMeta: any,
    config: MapChartConfig
  ) => {
    const lon = measureGetter(config, KEY.LONGTITUDE)!;
    const lat = measureGetter(config, KEY.LATITUDE)!;
    const color = measureGetter(config, KEY.COLOR);
    const pointExpr = {
      type: 'st_distance',
      fromlon: center.lng,
      fromlat: center.lat,
      tolon: lon.value,
      tolat: lat.value,
      distance: pointCircleR * 1000,
    };
    const filters = dataMeta && dataMeta.params.sql.match(/.*SELECT.*WHERE(.*)LIMIT .*\) as.*$/);
    let AND = '';
    if (filters && filters.length === 2) {
      AND = `AND (${filters[1]})`;
    }
    const {measures = []} = config;
    const columns = measures
      .filter((item: any) => item.isCustom)
      .map((item: any) => `${item.value} as ${item.as}`)
      .join(', ');
    const pointSql = `select * ${columns && `,${columns}`} from ${
      config.source
    } where ${parseExpression(pointExpr)} ${AND} ${
      color ? `ORDER BY ${color.value} DESC` : ''
    } limit 1`;
    return pointSql;
  };
  const onMouseMove = ({e, map, container}: any) => {
    pointRequest.current = delayRunFunc(
      {e, map, container},
      ({e, map, container}: any) => {
        const {pointSize} = config;
        const center = {lng: e.lngLat.lng, lat: e.lngLat.lat};
        const pointCircleR = _disPerPixelGetter(map, container, distance) * pointSize!;
        const pointSql = _genPointMapPointSql(center, pointCircleR, dataMeta, config);
        getRowBySql(pointSql).then(
          (rows: any) => {
            console.info(rows);
            if (rows && rows.length) {
              const html =
                popupContentGetter(config, rows[0]) +
                `<button class="view-all" data-id="view-all">${nls.label_view_all}</button>`;
              // create marker
              let el = document.createElement('div');
              el.className = 'marker';
              el.style.backgroundColor = getPointColor(config, rows[0]);
              el.style.cursor = 'initial';
              const size = '14px';
              el.style.width = size;
              el.style.height = size;
              if (popup.current) {
                popup.current.remove();
              }
              if (marker.current) {
                marker.current.remove();
              }
              // get marker pos
              let markerPos = markerPosGetter(config, rows[0], center);
              marker.current = new mapboxgl.Marker(el).setLngLat(markerPos).addTo(map);
              // create pop up
              popup.current = new mapboxgl.Popup({offset: 20, maxWidth: '600px'})
                .setLngLat(markerPos)
                .setHTML(html)
                .addTo(map);

              const popupEl = popup.current.getElement();
              const _onClicked = (e: any) => {
                _onPopupClicked(e, rows[0], popup.current);
              };
              if (popupEl) {
                popupEl.addEventListener('click', _onClicked);
              }
              // bind popup event
              popup.current.on('close', () => {
                popupEl.removeEventListener('click', _onClicked);
                popup.current = null;
                if (marker.current) {
                  marker.current.remove();
                  marker.current = null;
                }
              });
            }
          },
          () => {}
        );
      },
      600
    );
  };
  const onMouseOut = () => {
    pointRequest.current && pointRequest.current();
  };
  const _onPopupClicked = (e: any, row: any, _: any) => {
    if (e.target && e.target.dataset && e.target.dataset.id !== 'view-all') {
      return;
    }

    if (paperRef.current && detailRef.current) {
      // show panel
      paperRef.current.classList.add('show');
      setLoading(true);
      setAllowPopUp(false);
      if (row) {
        setLoading(false);
        detailRef.current.innerHTML = popupContentBuilder(config, row);
      }
    }
  };
  const onClose = () => {
    if (paperRef.current) {
      paperRef.current.classList.remove('show');
      setLoading(false);
      setAllowPopUp(true);
    }
  };

  return (
    <div
      className="z-map-chart z-point-map-chart"
      style={{
        background: theme.palette.background.paper,
      }}
    >
      <MapboxGl
        {...props}
        onMapUpdate={onMapUpdate}
        onDrawUpdate={onDrawUpdate}
        // onMouseMove={onMouseMove}
        onMouseOut={onMouseOut}
        draws={config.draws || []}
        allowPopUp={allowPopUp}
      />
      <Paper className="z-map-info-container" ref={paperRef}>
        {isLoading && (
          <div className="loading-container">
            <Spinner />
          </div>
        )}

        <div ref={detailRef}></div>
        <button className="close-button" onClick={onClose} type="button">
          {nls.label_close}
        </button>
      </Paper>
    </div>
  );
};

const getPointColor = (config: WidgetConfig, row: any): string => {
  const {colorKey = DEFAULT_COLOR, colorItems = []} = config;
  const colorMeasure = measureGetter(config, 'color');
  if (!colorMeasure) {
    return colorKey;
  }
  const isGradientRange = isGradientType(colorKey);
  const isDistinctColorRange = colorItems;
  const value = row[colorMeasure.isCustom ? colorMeasure.as : colorMeasure.value];
  const getColor = genColorGetter(config);
  if (isGradientRange) {
    return getColor(value);
  }
  if (isDistinctColorRange) {
    const colorItem = colorItems.filter((s: ColorItem) => s.as === row[colorMeasure.value])[0];
    if (colorItem) {
      return colorItem.color;
    }
    return DEFAULT_COLOR;
  }
  return colorKey;
};
export default PointMapNormal;
