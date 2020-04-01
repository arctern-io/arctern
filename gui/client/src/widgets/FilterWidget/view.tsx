import React, {FC, useEffect} from 'react';
import {CONFIG} from '../../utils/Consts';
import FilterWidget from '.';
import {FilterWidgetViewProps, Item, Option, Value} from './types';
import {Dimension} from '../../types';
import {getColType} from '../../utils/ColTypes';
import {formatterGetter} from '../../utils/Formatters';
import {getNumSeconds, TimeBin} from '../../utils/Time';
import {handleFilter, isSelected as _isSelected} from '../Utils/filters/common';

const FilterWidgetView: FC<FilterWidgetViewProps> = props => {
  const {config, setConfig, wrapperWidth, wrapperHeight} = props;

  const items = config.dimensions.map((d: Dimension) => {
    const {value, as, type} = d;
    const formatter = formatterGetter(d);
    let _options: any = [];
    let options = [];
    switch (getColType(type)) {
      case 'text':
        _options = (d.options || []).map((opt: string) => {
          return {value: opt, isSelected: _isSelected(opt, as, config), label: opt};
        });
        break;
      case 'number':
        const {maxbins = 2, extent = []} = d;
        const [min, max] = extent as number[];
        const stepRange = ((max as number) - (min as number)) / maxbins;
        for (let i = 0; i < maxbins; i++) {
          options.push([min + stepRange * i, min + stepRange * (i + 1)]);
        }
        _options = options.map((opt: number[]) => {
          const label = opt.map((o: number) => formatter(o)).join(' ~ ');
          const isSelected = _isSelected(opt, as, config);
          return {value: opt[0], label, isSelected};
        });
        break;
      case 'date':
        const {timeBin, extract, min: currMin, max: currMax} = d;
        if (extract) {
          for (let i = 1; i < (currMax as number); i++) {
            options.push(i);
          }
          _options = options.map((opt: number) => {
            return {value: opt, isSelected: _isSelected(opt, as, config), label: opt};
          });
        } else {
          const min = new Date(currMin as string).getTime();
          const max = new Date(currMax as string).getTime();
          const numSeconds = getNumSeconds(timeBin as TimeBin);
          const groups = (max - min) / numSeconds / 1000;
          for (let i = 0; i < groups; i++) {
            options.push([
              new Date(min + numSeconds * 1000 * i),
              new Date(min + numSeconds * 1000 * (i + 1)),
            ]);
          }
          _options = options.map((opt: Date[]) => {
            const label = opt.map((o: Date) => formatter(o)).join(' ~ ');
            const isSelected = _isSelected(opt[0], as, config);
            return {value: opt[0], label, isSelected};
          });
        }
        break;
      default:
        break;
    }
    return {
      value,
      as,
      options: _options,
    };
  });

  useEffect(() => {
    const selectedValues = items
      .map((item: Item) => item.options)
      .flat(Infinity)
      .filter((item: Option) => item.isSelected)
      .map((item: Option) => item.value);
    Object.keys(config.filter).forEach((key: string) => {
      const filter = config.filter[key];
      if (filter.expr.type === 'in') {
        const set = filter.expr.set;
        const isValid = set.some((val: Value) => {
          return !!selectedValues.find((selectedValue: Value) => selectedValue === val);
        });
        !isValid && setConfig({type: CONFIG.DEL_FILTER, payload: {filterKeys: [key]}});
      }
      if (filter.expr.type === 'between') {
        const {left, right} = filter.expr;
        const isValid = selectedValues.some((selectedValue: Value) => {
          if (!Array.isArray(selectedValue)) {
            return (
              left ===
              (selectedValue ||
                (left === (selectedValue as number) + 1 && right === (selectedValue as number) + 2))
            );
          }
          const isNumMatch = selectedValue[0] === left && selectedValue[1] === right;
          const isDateMatch =
            left === new Date(selectedValue[0]).toUTCString() &&
            right === new Date(selectedValue[1]).toUTCString();
          return isNumMatch || isDateMatch;
        });
        !isValid && setConfig({type: CONFIG.DEL_FILTER, payload: {filterKeys: [key]}});
      }
    });
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(config.dimensions)]);

  return (
    <FilterWidget
      wrapperWidth={wrapperWidth}
      wrapperHeight={wrapperHeight}
      items={items}
      onClick={(val: string, as: string) => handleFilter({val, as, config, setConfig})}
    />
  );
};

export default FilterWidgetView;
