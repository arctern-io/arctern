import React, {FC, useState, useEffect, useRef, useContext} from 'react';
import {dateFormat} from '../../utils/Formatters';
import {dimensionGetter, brushDataGetter} from '../../utils/WidgetHelpers';
import {TIME_GAPS} from '../../utils/Time';
import {I18nContext} from '../../contexts/I18nContext';
import {WidgetConfig} from '../../types';

// ["2009-04-08T00:00:00", "2009-04-13T00:00:00"]
// "1d"
// Fri May 01 2009 00:44:00 GMT+0800 (中国标准时间)
const nextBrushGetter = (brush: any[], timeUnit: string, max: any) => {
  const timeGap: any = TIME_GAPS[timeUnit];
  const timeMax = Date.parse(max);

  const nextBrush = brush.map((b: any) => {
    const next = Date.parse(b) + timeGap;
    return dateFormat(next > timeMax ? timeMax : next);
  });
  return nextBrush;
};

const Timeline_PLAY_TIMEOUT = 1500;

interface ITimelinePlayerProps {
  config: WidgetConfig;
  x: any;
  onPlay: ([a, b]: [any, any]) => void;
  isTimeChart: boolean;
}

const TimelinePlayer: FC<ITimelinePlayerProps> = props => {
  const {nls} = useContext(I18nContext);
  const {config, x, onPlay, isTimeChart} = props;
  const [playing, setPlaying] = useState(false);
  const playingTimeout = useRef<any>(null);

  const xDimension = dimensionGetter(config, 'x')!;
  const {binningResolution} = xDimension;
  const currMax = x.domain()[1];
  const timeMax = dateFormat(currMax);
  const {brush, showBrush} = brushDataGetter(xDimension, config.filter, isTimeChart);
  const showPlayer = isTimeChart && showBrush;
  const [nextBrushleft, nextBrushRight] = nextBrushGetter(brush, binningResolution!, currMax);
  const diffNow = Math.abs(Date.parse(brush[0]) - Date.parse(brush[1]));
  const diffNext = Math.abs(Date.parse(nextBrushleft) - Date.parse(nextBrushRight));
  const shouldStop = nextBrushRight === timeMax && diffNext < diffNow;

  useEffect(() => {
    if (!showPlayer || !playing) {
      clearTimeout(playingTimeout.current);
      setPlaying(false);
      return;
    }

    if (shouldStop) {
      setPlaying(false);
      return;
    }

    const play: any = () => {
      return setTimeout(() => {
        onPlay([nextBrushleft, nextBrushRight]);
      }, Timeline_PLAY_TIMEOUT);
    };

    if (playing) {
      clearTimeout(playingTimeout.current);
      playingTimeout.current = play();
    }

    return () => {
      clearTimeout(playingTimeout.current);
    };
  }, [playing, onPlay, showPlayer, nextBrushleft, nextBrushRight, shouldStop]);

  const onClick = () => {
    const action = !playing;
    if (!action) {
      clearTimeout(playingTimeout.current);
    }
    setPlaying(action);
  };

  return (
    <div className={`z-timeline-player ${showPlayer ? '' : 'hidden'}`}>
      <button
        title={playing ? nls.label_pause : nls.label_play}
        className={playing ? 'playing' : 'pause'}
        onClick={onClick}
      />
    </div>
  );
};

export default TimelinePlayer;
