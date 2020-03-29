import React, {useEffect, FC, useRef, useContext} from 'react';
import {useTheme} from '@material-ui/core/styles';
import Paper from '@material-ui/core/Paper';
import mapboxgl from 'mapbox-gl';
import MapboxGl from '../common/MapboxGl';
import {queryContext} from '../../contexts/QueryContext';
import {mapUpdateConfigHandler, drawUpdateConfigHandler} from '../Utils/filters/map';
import {CONFIG} from '../../utils/Consts';
import {formatterGetter} from '../../utils/Formatters';
import {delayRunFunc} from '../../utils/EditorHelper';
import {dimensionGetter, measureGetter} from '../../utils/WidgetHelpers';
import {id as genID} from '../../utils/Helpers';
import {ChoroplethMapProps} from './types';
const wkx = require('wkx');

const ChoroplethMapView: FC<ChoroplethMapProps> = props => {
  const theme = useTheme();
  const {getRowBySql, generalRequest} = useContext(queryContext);
  const {config, setConfig} = props;
  const paperRef = useRef<any>();
  const detailRef = useRef<any>();
  const pointRequest = useRef<any>(null);
  const popup = useRef<any>(null);

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
  const onDrawUpdate = (draws: any) => {
    setConfig({
      type: CONFIG.UPDATE,
      payload: drawUpdateConfigHandler(config, draws),
    });
  };
  const _preSql = (center: any) => {
    const data = {
      data: [
        {
          name: 'render_type',
          values: ['get_building_shape'],
        },
        {
          name: 'cursor_position',
          values: [center.lng, center.lat],
        },
      ],
    };
    return JSON.stringify(data);
  };
  const _genChoroplethMapPointSql = ({config, center}: any) => {
    const preFix = `pixel_get_building_shape('${_preSql(center)}')`;
    const pgSql = `SELECT ${preFix} FROM ${config.source}`;
    return pgSql;
  };
  const _pointValSqlGetter = (config: any, buildVal: string) => {
    const buildDimension = dimensionGetter(config, 'build')!;
    const lonMeasure = measureGetter(config, 'lon')!;
    const latMeasure = measureGetter(config, 'lat')!;
    const colorMeasure = measureGetter(config, 'color')!;

    const sql = `SELECT AVG(${lonMeasure.value}) as ${lonMeasure.as}, AVG(${latMeasure.value}) as ${
      latMeasure.as
    }, ${colorMeasure.isCustom ? '' : colorMeasure.expression}(${colorMeasure.value}) as ${
      colorMeasure.as
    } FROM ${config.source} WHERE (${buildDimension.value} IN ('${buildVal}')) GROUP BY ${
      buildDimension.value
    }`;
    return sql;
  };
  const _cleanLayer = ({map, id}: any) => {
    if (map && map.getLayer && map.getLayer(id)) {
      map.removeLayer(id);
    }
  };
  const _cleanSource = ({map, id}: any) => {
    if (map && map.getSource && map.getSource(id)) {
      map.removeSource(id);
    }
  };
  const _cleanMap = ({map, id}: any) => {
    _cleanLayer({map, id});
    _cleanSource({map, id});
  };
  const _parseCoordinate = (data: string) => {
    const coordinates = wkx.Geometry.parse(data);
    const {exteriorRing = [], interiorRings = []} = coordinates;
    return {
      exter: _parseRing(exteriorRing),
      inter: _parseRing(interiorRings),
    };
  };
  const _addLayer = ({map, coordinates, id}: any) => {
    _cleanMap({map, id});
    map.addLayer({
      id,
      type: 'line',
      source: {
        type: 'geojson',
        data: {
          type: 'Feature',
          geometry: {
            type: 'Polygon',
            coordinates: [coordinates],
          },
        },
      },
      layout: {},
      paint: {
        'line-color': '#fff',
        'line-opacity': 0.8,
        'line-width': 2,
      },
    });
  };
  const _parseRing = (ring: any[]) => ring.map((r: any) => [r.x, r.y]);
  const _addChoroLayer = ({map, coordinatesGrp}: any) => {
    const {exter, inter} = coordinatesGrp;
    _addLayer({map, coordinates: exter, id: 'hover-exter'});
    _addLayer({map, coordinates: inter, id: 'hover-inter'});
  };
  const onMouseMove = ({e, map}: any) => {
    pointRequest.current = delayRunFunc(
      {e, map},
      ({e, map}: any) => {
        const center = {lng: e.lngLat.lng, lat: e.lngLat.lat};
        const pointSql = _genChoroplethMapPointSql({config, center});
        getRowBySql(pointSql).then((rows: any) => {
          if (rows && rows.length) {
            const buildVal = rows[0].pixel_get_building_shape;
            if (buildVal) {
              const pointSql = _pointValSqlGetter(config, buildVal);
              generalRequest({id: genID(), sql: pointSql}).then((res: any) => {
                if (res && res[0]) {
                  const data = res[0];
                  const html = popupContentGetter(config, data);
                  // create marker
                  let el = document.createElement('div');
                  el.className = 'marker';
                  el.style.cursor = 'initial';
                  const size = '14px';
                  el.style.width = size;
                  el.style.height = size;
                  // clear first
                  if (popup.current) {
                    popup.current.remove();
                  }
                  // get  pos
                  let popupPos = {lat: data.lat, lng: data.lon};
                  popup.current = new mapboxgl.Popup({offset: 20, maxWidth: '600px'})
                    .setLngLat(popupPos)
                    .setHTML(html)
                    .addTo(map);
                  // bind popup event
                  popup.current.on('close', () => {
                    _cleanMap({map, id: 'hover-exter'});
                    _cleanMap({map, id: 'hover-inter'});
                    popup.current = null;
                  });
                  // Draw layer
                  const coordinatesGrp = _parseCoordinate(buildVal);
                  _addChoroLayer({map, coordinatesGrp});
                }
              });
            }
          }
        });
      },
      600
    );
  };
  const onMouseOut = () => {
    pointRequest.current && pointRequest.current();
  };
  const popupContentGetter = (config: any, row: any) => {
    let content = `<ul>`;
    config.measures.forEach((m: any) => {
      const {value, as} = m;
      const formatter = formatterGetter(m);
      content += `<li><span class="content-title"><strong>${value}:</strong></span><span>${formatter(
        row[as]
      )}</span></li>`;
    });

    return content + `</ul>`;
  };
  useEffect(() => {
    return () => {
      popup.current && popup.current.remove();
    };
  }, []);
  return (
    <div
      className="z-map-chart"
      style={{
        backgroundColor: theme.palette.background.paper,
      }}
    >
      <MapboxGl
        {...props}
        onMapUpdate={onMapUpdate}
        onDrawUpdate={onDrawUpdate}
        onMouseMove={onMouseMove}
        onMouseOut={onMouseOut}
        draws={config.draws || []}
        allowPopUp={true}
        showRuler={false}
      />
      <Paper className="z-map-info-container" ref={paperRef}>
        <div ref={detailRef}></div>
      </Paper>
    </div>
  );
};

export default ChoroplethMapView;
