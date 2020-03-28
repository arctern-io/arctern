import React, {FC} from 'react';
import {useTheme} from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import {FilterWidgetProps, Item, Option} from './types';
import './style.scss';

const FilterWidget: FC<FilterWidgetProps> = props => {
  const {items, wrapperWidth, wrapperHeight, onClick} = props;
  const theme = useTheme();
  return (
    <div
      className="z-filter-chart"
      style={{
        width: wrapperWidth,
        height: wrapperHeight,
        background: theme.palette.background.paper,
      }}
    >
      {items.map((opt: Item) => {
        const {options, as, value} = opt;
        return (
          <div key={as} className="filter-wrapper">
            <h3>{value}</h3>
            <div className="option">
              {options.map((opt: Option) => {
                const {isSelected, value, label} = opt;
                return (
                  <div key={value.toString()}>
                    <Button
                      onClick={() => onClick(value, as)}
                      color={isSelected ? 'primary' : 'default'}
                    >
                      {label}
                    </Button>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default FilterWidget;
