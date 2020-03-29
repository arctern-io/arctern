import React, {FC, useRef, useEffect} from 'react';
import {useTheme} from '@material-ui/core/styles';
import {DefaultWidgetProps} from '../../types';
import {measureGetter} from '../../utils/WidgetHelpers';
import {genColorGetter} from '../../utils/Colors';

declare global {
  interface Window {
    echarts: any;
  }
}

const ScatterChartView: FC<DefaultWidgetProps> = props => {
  const theme = useTheme();
  const {data, wrapperWidth, wrapperHeight, config} = props;
  const container = useRef<HTMLDivElement>(null);
  const chartRef = useRef<any>(null);
  const getColor = genColorGetter(config);

  const x = measureGetter(config, 'x')!.value;
  const y = measureGetter(config, 'y')!.value;
  const z = measureGetter(config, 'z')!.value;

  const style = {
    width: wrapperWidth,
    height: wrapperHeight,
    background: theme.palette.background.paper,
  };

  const effectFactors = JSON.stringify([
    data,
    wrapperWidth,
    wrapperHeight,
    x,
    y,
    z,
    config.colorKey,
    config.ruler,
    config.colorItems,
  ]);

  useEffect(() => {
    if (!chartRef.current) {
      chartRef.current = window.echarts.init(container.current, 'dark');
    }
    const convertData = (data: any[]) => {
      const _data = data.map((d: any) => {
        return [d.x, d.y, d.z, d.size, d.color];
      });
      return _data;
    };

    const sourceData = convertData(data);

    if (chartRef.current) {
      chartRef.current.setOption({
        grid3D: {
          axisLine: {
            lineStyle: {
              color: '#000',
            },
          },
          axisPointer: {
            lineStyle: {
              color: '#ffbd67',
            },
          },
          viewControl: {
            rotateSensitivity: 1,
          },
        },
        xAxis3D: {name: x, color: '#FFF'},
        yAxis3D: {name: y},
        zAxis3D: {name: z},
        dataset: {
          source: sourceData,
        },
        series: [
          {
            type: 'scatter3D',
            symbolSize: (data: any) => {
              return data[3] || 2.5;
            },
            encode: {
              tooltip: [0, 1, 2],
            },
            itemStyle: {
              color: (selected: any) => {
                return getColor(selected.data[4], theme.palette.background.default);
              },
            },
          },
        ],
      });
      chartRef.current.resize();
    }
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [effectFactors]);

  return <div ref={container} style={style}></div>;
};

export default ScatterChartView;
