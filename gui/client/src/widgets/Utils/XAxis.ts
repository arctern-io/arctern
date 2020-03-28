import {scaleLinear, scaleTime, scaleBand, axisBottom, select} from 'd3';
import {COLUMN_TYPE} from '../../utils/Consts';
import {formatterGetter} from '../../utils/Formatters';
import {EXTRACT_INPUT_OPTIONS} from '../../utils/Time';
import {dimensionTypeGetter} from '../../utils/WidgetHelpers';

class XAxis {
  x: any;
  xAxis: any;
  type: COLUMN_TYPE;
  formatter: Function;

  constructor(xDimension: any, width: number) {
    const xTime = scaleTime()
      .range([0, width])
      .nice();
    const xLinear = scaleLinear()
      .rangeRound([0, width])
      .nice();
    const xBand = scaleBand().range([0, width]);
    this.type = dimensionTypeGetter(xDimension);
    this.x =
      this.type === COLUMN_TYPE.DATE ? xTime : this.type === COLUMN_TYPE.NUMBER ? xLinear : xBand;
    this.formatter = formatterGetter(xDimension, 'axis');
  }

  // add invert function for text type
  addInvert() {
    if (this.type === COLUMN_TYPE.TEXT && !this.x.invert) {
      this.x.invert = (v: number): string => {
        const eachBand = this.x.step();
        const i = Math.round(v / eachBand);
        return this.x.domain()[i] || this.x.domain()[i - 1];
      };
    }
  }

  // get d3 x instance
  get xScale() {
    this.addInvert();
    return this.x;
  }

  // get formatter for different data type
  get defaultFormatter() {
    return this.formatter;
  }

  setDomain(data: []) {
    this.x.domain(data);
    this.xAxis = axisBottom(this.x);
  }

  setTicks(tick: number, xDimension: any, formatter?: Function) {
    const tickFormatter = formatter || this.formatter;
    this.xAxis.tickFormat(tickFormatter);
    this.xAxis.ticks(tick);
    if (xDimension.extract && xDimension.isBinned) {
      let opt =
        EXTRACT_INPUT_OPTIONS.filter((item: any) => item.value === xDimension.timeBin)[0] || {};
      let numTicks = opt.max || 100;
      this.xAxis.ticks(numTicks);
    }
  }

  update(xAxisDom: SVGGElement) {
    select(xAxisDom)
      .transition()
      .call(this.xAxis);
  }
}

export default XAxis;
