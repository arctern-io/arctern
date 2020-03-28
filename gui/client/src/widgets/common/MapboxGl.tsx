/// <reference path="Mapbox.d.ts" />
import React, {FC, useState, useEffect, useRef, useContext} from 'react';
import mapboxgl from 'mapbox-gl';
import {MapChartProps} from './MapChart.type';
import {
  TRANSPARENT_PNG,
  DEFAULT_ZOOM,
  defaultMapCenterGetter,
  mapboxCoordinatesGetter,
  onMapLoaded as defaultOnMapLoaded,
} from '../Utils/Map';
import {I18nContext} from '../../contexts/I18nContext';
import {rootContext} from '../../contexts/RootContext';
import {queryContext} from '../../contexts/QueryContext';
import {DEFAULT_WIDGET_WRAPPER_WIDTH, DEFAULT_WIDGET_WRAPPER_HEIGHT} from '../../utils/Layout';
import MapboxGlDraw from './MapboxGlDraw';
import GradientRuler from './GradientRuler';
import './MapboxGl.scss';

const MapboxLanguage = require('@mapbox/mapbox-gl-language');
const ZoomControl = require('mapbox-gl-controls/lib/zoom').default;

const MapboxGl: FC<MapChartProps> = props => {
  const {language} = useContext(I18nContext);
  const {globalConfig} = useContext(rootContext);
  const {getMapBound} = useContext(queryContext);
  mapboxgl.accessToken = globalConfig.MAPBOX_ACCESS_TOKEN; // eslint-disable-line
  const {
    config,
    setConfig,
    onMapLoaded = defaultOnMapLoaded,
    onMapUpdate,
    onDrawUpdate,
    onMouseMove,
    onMouseOut,
    draws,
    wrapperWidth = DEFAULT_WIDGET_WRAPPER_WIDTH,
    wrapperHeight = DEFAULT_WIDGET_WRAPPER_HEIGHT,
    allowPopUp = false,
  } = props;
  const [map, setMap] = useState<any>(null);
  const [mapLoaded, setMaploaded] = useState(false);
  const mapContainer = useRef<any>();
  const imageSourceCache = useRef<any>(null);
  const mapMoveTimeout = useRef<any>(null);
  const mapResizeTimeout = useRef<any>(null);
  const movingThrottle = 350;
  let data = Array.isArray(props.data) ? props.data[0] : `data:image/png;base64,${props.data}`;
  // create map
  useEffect(() => {
    const mapbox = new mapboxgl.Map({
      container: mapContainer.current,
      style: config.mapTheme,
      zoom: config.zoom || DEFAULT_ZOOM,
      center: config.center || defaultMapCenterGetter(language),
    });
    const _language = new MapboxLanguage();
    mapbox.addControl(_language);
    mapbox.addControl(new ZoomControl(), 'bottom-right');

    const onStyleLoad = () => {
      // add init layer
      const bounds = mapbox.getBounds();
      const coordinates = mapboxCoordinatesGetter(bounds);
      // add image source
      const imageSource = mapbox.addSource('overlaySource', {
        type: 'image',
        url: data || TRANSPARENT_PNG,
        coordinates,
      });

      let layers: any = mapbox.getStyle().layers;
      // Find the index of the first symbol layer in the map style
      let firstSymbolId;
      for (let i = 0; i < layers.length; i++) {
        if (layers[i].type === 'symbol') {
          firstSymbolId = layers[i].id;
          break;
        }
      }
      // add layer
      mapbox.addLayer(
        {
          id: 'overlay',
          source: 'overlaySource',
          type: 'raster',
          paint: {
            'raster-opacity': 1,
            'raster-fade-duration': 0,
          },
        },
        firstSymbolId
      );
      // set local cache
      imageSourceCache.current = imageSource;
    };
    // mapbox can call addLayer only after style.load
    mapbox.on('style.load', onStyleLoad);
    // refetch image on load
    // fix image not matched on first loading
    const onLoad = () => {
      if (mapContainer.current) {
        onMapLoaded(config, getMapBound).then((bounds: any) => {
          if (bounds !== -1) {
            mapbox.fitBounds(bounds);
          }
          onMapUpdate && onMapUpdate(mapbox, mapContainer.current);
        });
      }
      setMaploaded(true);
    };
    mapbox.on('load', onLoad);
    // update local state
    setMap(mapbox);
    return () => {
      // clear mapbox, avoid memory leak
      // eslint-disable-next-line
      mapbox.remove();
    };

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [config.mapTheme]);

  // pointer selection
  useEffect(() => {
    if (!map) {
      return;
    }
    const _onMouseMove = (e: any) =>
      onMouseMove && onMouseMove({e, map, container: mapContainer.current});
    if (allowPopUp) {
      map.on('mousemove', _onMouseMove);
      map.on('mouseout', onMouseOut);
    }
    return () => {
      map.off('mousemove', onMouseMove);
      map.off('mouseout', onMouseOut);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, allowPopUp]);

  useEffect(() => {
    if (!map) {
      return;
    }
    // on move, update bounds and zoom
    const onMove = () => {
      if (mapMoveTimeout.current) {
        clearTimeout(mapMoveTimeout.current);
      }
      mapMoveTimeout.current = setTimeout(() => {
        if (mapContainer.current) {
          onMapUpdate && onMapUpdate(map, mapContainer.current);
        }
      }, movingThrottle);
    };

    map.once('moveend', onMove);
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [map, onMapUpdate]);

  // image overlayer updater
  useEffect(() => {
    if (!map) {
      return;
    }
    let d = typeof data !== 'string' ? TRANSPARENT_PNG : data;
    if (imageSourceCache.current) {
      const bounds = map.getBounds();
      const overlayName = 'overlaySource';
      const imageSrc = map.getSource(overlayName);
      const coordinates = mapboxCoordinatesGetter(bounds);
      imageSrc &&
        imageSrc.updateImage({
          url: d,
          coordinates: coordinates,
        });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data, mapLoaded]);

  // resize
  useEffect(() => {
    if (!map) {
      return;
    }
    if (mapResizeTimeout.current) {
      clearTimeout(mapResizeTimeout.current);
    }

    const onMapResize = () => {
      if (mapResizeTimeout.current) {
        clearTimeout(mapResizeTimeout.current);
      }
      mapResizeTimeout.current = setTimeout(() => {
        onMapUpdate && onMapUpdate(map, mapContainer.current);
      }, movingThrottle);
    };
    map.off('resize', onMapResize);
    map.on('resize', onMapResize);

    if (map) {
      map.resize();
    }

    return () => {
      map.off('resize', onMapResize);
      clearTimeout(mapResizeTimeout.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wrapperHeight, wrapperWidth]);

  // render
  return (
    <div className="z-chart mapboxgl">
      <div ref={mapContainer} className="container" style={{height: '100%', width: '100%'}} />
      {map && (
        <>
          <MapboxGlDraw map={map} draws={draws} onDrawUpdate={onDrawUpdate} />
          <GradientRuler config={config} setConfig={setConfig} />
        </>
      )}
    </div>
  );
};

export default MapboxGl;
