import React, {FC, useRef, useEffect} from 'react';
import {select, event, brushX} from 'd3';
import {COLUMN_TYPE} from '../../utils/Consts';
import {typeNormalizeObjGetter} from '../../utils/WidgetHelpers';
import {brushDataGetter} from '../../utils/WidgetHelpers';

const BrushMoveDuration = 100;
const Brush: FC<any> = props => {
  const {
    x,
    dataMeta,
    linkMeta,
    config,
    filter,
    onBrushUpdate,
    width,
    height,
    isTimeChart,
    xDimension,
    onBrush = () => {},
    onBrushMove = () => {},
    brushMoveDuration = BrushMoveDuration,
  } = props;
  const brushNode: any = useRef();
  let isLoading = dataMeta && dataMeta.loading;

  if (linkMeta) {
    isLoading = linkMeta.loading;
  }

  const effectors = [
    config.layout,
    config.colorItems,
    width,
    height,
    x.domain(),
    isTimeChart,
    filter,
    isLoading,
  ];
  useEffect(() => {
    if (isLoading || !x.domain().length) {
      return;
    }

    let silent = false;
    // get brush data
    const {brush, showBrush, isTimeBrush, timeRound} = brushDataGetter(
      xDimension,
      filter,
      isTimeChart
    );
    const normalizer = typeNormalizeObjGetter(isTimeBrush ? COLUMN_TYPE.DATE : COLUMN_TYPE.NUMBER);
    const formattedBrushData = brush.map(normalizer);
    const initBrushData: any = formattedBrushData.map(x);
    // init brush
    const _onBrush = () => {
      if (!event.sourceEvent) return; // Only transition after input.
      if (event.sourceEvent.type === 'brush') return; // only handle brush
      if (!event.selection) return;
      let [xMin, xMax] = x.domain();
      let d0 = event.selection.map(x.invert);
      let d1: any = d0;
      if (isTimeBrush) {
        d1 = d0.map((d: any) => {
          if (d.getTime() === xMin.getTime() || d.getTime() === xMax.getTime()) {
            return d;
          } else {
            return timeRound(d);
          }
        });
      }
      // set boundary
      d1[0] = d1[0] <= xMin ? xMin : d1[0];
      d1[1] = d1[1] > xMax ? xMax : d1[1];
      select(brushNode.current).call(event.target.move, d1.map(x));
      onBrushMove(d1);
      if (isTimeBrush) {
        onBrush(d1);
      }
    };

    const _onBrushEnd = () => {
      let clear: boolean = !event.selection || event.selection[0] === event.selection[1];
      if (!silent) {
        onBrushUpdate(
          clear
            ? []
            : event.selection.map((v: any) =>
                xDimension.extract ? Math.round(x.invert(v)) : x.invert(v)
              )
        );
      }
      silent = false;
    };
    // bind event
    const brusher = brushX()
      .extent([
        [0, 0],
        [width, height],
      ])
      .on('brush', _onBrush)
      .on('end', _onBrushEnd);
    // init brush
    const brushNodeSelection: any = select(brushNode.current);
    brushNodeSelection.call(brusher);

    // set brush
    silent = true;
    brusher.move(
      brushNodeSelection.transition().duration(brushMoveDuration),
      showBrush ? initBrushData : null
    );

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(effectors)]);
  return <g ref={brushNode} className="brush" fill="none" pointerEvents="all" />;
};

export default Brush;
