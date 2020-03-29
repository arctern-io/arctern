import React, {FC, useEffect, useRef} from 'react';
import MapboxDraw from '@mapbox/mapbox-gl-draw';
const {DragCircleMode, DirectMode, SimpleSelectMode} = require('mapbox-gl-draw-circle');

const DRAW_OPTIONS = {
  drawing: true,
  boxSelect: true,
  keybindings: true,
  controls: {
    point: false,
    line_string: false,
    polygon: false,
    trash: false,
    circle: false,
    combine_features: false,
    uncombine_features: false,
  },
  userProperties: true,
  modes: {
    ...MapboxDraw.modes,
    drag_circle: DragCircleMode,
    direct_select: DirectMode,
    simple_select: SimpleSelectMode,
  },
};

const featuresGetter = (drawTools: any) => {
  return drawTools
    .getAll()
    .features.filter((f: any) => {
      if (f.properties.isCircle) {
        return f.properties.center.length > 0;
      }
      return f.geometry && f.geometry.coordinates && f.geometry.coordinates.some((c: any) => c);
    })
    .map((f: any) => ({
      type: f.geometry.type,
      data: f,
      id: f.id,
    }));
};

const MapboxGlDraw: FC<any> = props => {
  const {map, onDrawUpdate, draws} = props;
  // console.info(draws);
  const drawToolCache = useRef<any>();

  const isFirstRun = useRef(true);
  const drawUpdaterCache = useRef(onDrawUpdate);
  drawUpdaterCache.current = onDrawUpdate;

  useEffect(() => {
    // console.log("map load draws is", draws);
    if (isFirstRun.current) {
      // create draw tools
      const drawTools: any = new MapboxDraw(DRAW_OPTIONS);
      drawToolCache.current = drawTools;
      map.addControl(drawTools);

      // update filter on draw.create
      map.on('draw.create', (e: any) => {
        drawUpdaterCache.current(featuresGetter(drawTools));
      });
      // update filter on draw.update
      map.on('draw.update', () => {
        drawUpdaterCache.current(featuresGetter(drawTools));
      });
      map.on('draw.selectionchange', () => {
        // force update on selection change
        setTimeout(() => {
          drawUpdaterCache.current(featuresGetter(drawTools));
        });
      });
      map.on('draw.modechange', (e: any) => {
        // console.log("draw.modechange", e.mode);
        // if (e.mode === "direct_select") {
        //   console.log("direct_select");
        // }
      });
      // update filter on draw.delete
      map.on('draw.delete', () => {
        drawUpdaterCache.current(featuresGetter(drawTools));
      });
      isFirstRun.current = false;
    }

    // add draws
    drawToolCache.current.deleteAll();
    draws.forEach((d: any) => {
      drawToolCache.current.add(d.data);
    });

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(draws.map((d: any) => d.id))]);

  const onCicleClicked = (e: any) => {
    drawToolCache.current.changeMode('drag_circle');
  };

  const onPolygonClicked = (e: any) => {
    drawToolCache.current.changeMode('draw_polygon');
  };

  const onDelClick = (e: any) => {
    let selected = drawToolCache.current.getSelected();
    if (selected) {
      selected.features.forEach((f: any) => {
        drawToolCache.current.delete(f.id);
        map.fire('draw.delete');
      });
    }
  };

  return (
    <div className="draw-tool">
      <button className="button-circle" onClick={onCicleClicked} title="Create a circle" />
      <button className="button-polygon" onClick={onPolygonClicked} title="Create a polygon" />
      <button className="button-trash" onClick={onDelClick} title="delete selection" />
    </div>
  );
};

export default MapboxGlDraw;
