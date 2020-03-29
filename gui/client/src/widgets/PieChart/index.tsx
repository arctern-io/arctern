import React, {FC, Fragment, useContext, useRef, useEffect} from 'react';
import {useTheme} from '@material-ui/core/styles';
import {arc, pie, select, interpolate} from 'd3';
import {default_duration} from '../../utils/Animate';
import {PieChartProps} from './types';
import {measureGetter, dimensionDataGetter} from '../../utils/WidgetHelpers';
import {rootContext} from '../../contexts/RootContext';
import {dimensionsDataToFilterExpr} from '../../utils/Filters';
import {pieLabelDecorator} from '../Utils/Decorators';
import {cloneObj} from '../../utils/Helpers';
import {genColorGetter} from '../../utils/Colors';
import {rangeFormatter, formatterGetter} from '../../utils/Formatters';
import {DEFAULT_WIDGET_WRAPPER_WIDTH, DEFAULT_WIDGET_WRAPPER_HEIGHT} from '../../utils/Layout';
import './style.scss';

const PieChart: FC<PieChartProps> = props => {
  const theme = useTheme();
  const {showTooltip, hideTooltip} = useContext(rootContext);
  const {
    config,
    onPieClick,
    data,
    dataMeta,
    wrapperWidth = DEFAULT_WIDGET_WRAPPER_WIDTH,
    wrapperHeight = DEFAULT_WIDGET_WRAPPER_HEIGHT,
  } = props;
  const paddingLeft = 30,
    paddingRight = 30,
    paddingTop = 30,
    paddingBottom = 30;
  const getColor = genColorGetter(config);
  const svgWidth = wrapperWidth - paddingLeft - paddingRight;
  const svgHeight = wrapperHeight - paddingTop - paddingBottom;
  const container = useRef<any>(null);
  // arc builder
  const outerRadius = Math.min(svgWidth, svgHeight) / 2 - 1;
  const arcBuilder = arc()
    .innerRadius(outerRadius * (1 - 0.61803398875)) // this use for change PieChartType
    .outerRadius(outerRadius);

  // get size measure
  const sizeMeasure = measureGetter(config, 'size')!;
  // pie data getter
  const d3PieDataGetter: Function = pie()
    .sort(null)
    .value((d: any) => d[sizeMeasure.as]);

  // pie data preFormat
  const pieDataPreFormat = (data: any[], filters: any): any => {
    return data.map((row: any) => {
      let newRow: any = cloneObj(row);
      let currentFilterExpr = dimensionsDataToFilterExpr(newRow.dimensionsData);
      let filterNames = Object.keys(filters);
      newRow.filters = [];
      newRow.selected =
        filterNames.length === 0
          ? true
          : filterNames.filter((f: any) => {
              if (filters[f].expr === currentFilterExpr) {
                newRow.filters.push(f);
                return true;
              } else {
                return false;
              }
            }).length > 0;
      return newRow;
    });
  };

  const pieData = d3PieDataGetter(
    pieDataPreFormat(dimensionDataGetter(config.dimensions, data), config.filter)
  );
  // pie radius
  const radius = (Math.min(svgWidth, svgHeight) / 2) * 0.8;
  const arcLabel = arc()
    .innerRadius(radius)
    .outerRadius(radius);

  // animation
  const pieContainer = useRef<any>(null);

  useEffect(() => {
    if (!dataMeta || dataMeta.loading) {
      return;
    }
    const pathElArr = Array.from(pieContainer.current.children || []).filter(
      (v: any) => v && v.nodeName === 'path'
    );
    pathElArr.forEach((v: any) => {
      const newPie = pieData.find((pie: any) => pie.index === Number(v.dataset.index));

      select(v)
        .transition()
        .duration(default_duration)
        .attrTween('d', () => {
          const i = interpolate(
            {
              startAngle: Number(v.dataset.startAngle),
              endAngle: Number(v.dataset.endAngle),
            },
            newPie
          );
          return function(t: any) {
            return arcBuilder(i(t)) as string;
          };
        })
        .attr('data-end-angle', newPie.endAngle)
        .attr('data-start-angle', newPie.startAngle);
    });
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dataMeta && dataMeta.loading, wrapperWidth, wrapperHeight, JSON.stringify(pieData)]);
  // event
  const onClick = (e: any) => {
    let index = e.target.dataset.index * 1;

    if (onPieClick) {
      onPieClick(pieData[index]);
    }
  };

  // Tooltip
  const tooltipTitleGetter = (tooltipData: any) => {
    return (
      <div>
        <span
          className="mark"
          style={{background: `${getColor(tooltipData.as) || theme.palette.background.default}`}}
        />
        {rangeFormatter(tooltipData.dimensionsData)}
      </div>
    );
  };

  const tooltipContentGetter = (tooltipData: any) => {
    return pieMeasureFormatter(tooltipData.data[sizeMeasure.as]);
  };

  const onMouseMove = (e: any) => {
    let index = e.target.dataset.index;
    let pie = pieData[index];
    let tooltipData = {
      x: e.clientX,
      xv: '',
      data: {...pie.data},
      as: pie.data[sizeMeasure.as] + '',
      dimensionsData: pie.data.dimensionsData,
    };
    let position = {
      event: e,
    };

    showTooltip({
      position,
      tooltipData,
      titleGetter: tooltipTitleGetter,
      contentGetter: tooltipContentGetter,
    });
  };

  const onMouseOut = () => {
    hideTooltip();
  };

  const pieMeasureFormatter = (v: any) => {
    return formatterGetter(sizeMeasure)(v);
  };
  return (
    <div
      className="z-chart z-pie-chart"
      ref={container}
      style={{
        background: theme.palette.background.paper,
      }}
    >
      <svg
        width={svgWidth}
        height={svgHeight}
        viewBox={`${-svgWidth / 2}, ${-svgHeight / 2}, ${svgWidth}, ${svgHeight}`}
      >
        <g textAnchor="middle" fontSize="12" fontFamily="sans-serif">
          <g ref={pieContainer}>
            {pieData.map((pie: any) => (
              <Fragment key={pie.index}>
                <path
                  className={`arc ${pie.data.selected ? '' : 'unselected'}`}
                  fill={getColor(pie.data[sizeMeasure.as] + '') as string}
                  stroke={getColor(pie.data[sizeMeasure.as] + '') as string}
                  style={{cursor: 'pointer'}}
                  d=""
                  data-start-angle={0}
                  data-end-angle={0}
                  data-index={pie.index}
                  onMouseMove={onMouseMove}
                  onMouseOut={onMouseOut}
                  onClick={onClick}
                />
                <text transform={`translate(${arcLabel.centroid(pie)})`} pointerEvents="none">
                  <tspan className="dimension" y="-0.4em" fontWeight="bold">
                    {pieLabelDecorator(
                      rangeFormatter(pie.data.dimensionsData),
                      pie,
                      svgWidth / 2,
                      arcLabel.centroid(pie)
                    )}
                  </tspan>
                  <tspan className="measure" x="0" y="0.7em" fillOpacity="0.7">
                    {pieLabelDecorator(
                      pieMeasureFormatter(pie.data[sizeMeasure.as]),
                      pie,
                      svgWidth / 2,
                      arcLabel.centroid(pie)
                    )}
                  </tspan>
                </text>
              </Fragment>
            ))}
          </g>
        </g>
      </svg>
    </div>
  );
};

export default PieChart;
