import React, {useContext, useRef} from 'react';
import {I18nContext} from '../../contexts/I18nContext';
import clsx from 'clsx';
import {makeStyles, useTheme} from '@material-ui/core/styles';
import {genBasicStyle} from '../../utils/Theme';
import {Measure} from '../../types';
import {measureGetter, dimensionGetter} from '../../utils/WidgetHelpers';
import {
  gradientOpts,
  color as calDefaultColor,
  UNSELECTED_COLOR,
  isGradientType,
} from '../../utils/Colors';
import {CONFIG} from '../../utils/Consts';
import {cloneObj} from '../../utils/Helpers';
import {formatterGetter} from '../../utils/Formatters';
import './GradientRuler.scss';

const useStyles = makeStyles(
  theme =>
    ({
      ...genBasicStyle(theme.palette.primary.main),
      root: {
        position: 'absolute',
        bottom: '10px',
        left: '15px',
        display: 'flex',
        justifyContent: 'space-between',
        fontSize: '9px',
      },
      distinctRoot: {
        position: 'absolute',
        bottom: '10px',
        left: '15px',
        display: 'flex',
        flexDirection: 'column',
        fontSize: '9px',
      },
      ruler: {
        display: 'flex',
        width: '100px',
        height: '35px',
      },
      rulerItemContainer: {
        width: '10px',
        height: '32px',
        overflow: 'visible',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        justifyContent: 'start',
      },
      rulerItem: {
        width: '10px',
        height: '10px',
        marginBottom: '2px',
      },
      colorItemContainer: {
        marginBottom: '5px',
        display: 'flex',
        justifyContent: 'first',
        alignItems: 'center',
      },
      colorItem: {
        width: '20px',
        height: '20px',
        marginRight: '10px',
      },
      customHover: {},
      rulerRange: {
        display: 'flex',
        justifyContent: 'space-between',
      },
    } as any)
) as Function;

const getShowType = (config: any) => {
  const useGradientColor = isGradientType(config.colorKey);
  const useDistinctColor = config.colorItems && config.colorItems.length > 0;
  if (useGradientColor) {
    return 'byGradientColor';
  }
  if (useDistinctColor) {
    return 'byDistinctColor';
  }
  return 'byMeasures';
};
const GradientRuler = (props: any) => {
  const {nls} = useContext(I18nContext);
  const theme = useTheme();
  const classes = useStyles(theme);
  const {config = {}, setConfig} = props;
  const cloneConfig = cloneObj(config);
  const {colorKey, ruler = {min: 5, max: 100}, colorItems = [], filter, id} = cloneConfig;
  const target = gradientOpts.find((item: any) => item.key === colorKey) || gradientOpts[0];
  const {value} = target;
  const rulerMin: any = useRef(null);
  const rulerMax: any = useRef(null);
  const onRulerEnter = (e: any) => e.currentTarget.classList.add('fade');
  const onRulerLeave = (e: any) => e.currentTarget.classList.remove('fade');
  const onRulerItemEnter = (e: any) => e.currentTarget.classList.add('show-item');
  const onRulerItemLeave = (e: any) => e.currentTarget.classList.remove('show-item');

  const upDateFilter = (filterValue: string) => {
    const colorMeasure = measureGetter(config, 'color')!;
    const {as, value} = colorMeasure;
    if (!filter[as]) {
      const newFilter = {
        [as]: {
          type: 'filter',
          expr: {
            type: 'in',
            set: [filterValue],
            expr: value,
          },
        },
      };
      setConfig({type: CONFIG.ADD_FILTER, payload: {...newFilter, id}});
      return;
    }
    const targetFilter = filter[as];
    const isExist = targetFilter.expr.set.some((val: string) => val === filterValue);
    if (isExist) {
      targetFilter.expr.set = targetFilter.expr.set.filter((val: string) => val !== filterValue);
      targetFilter.expr.set.length === 0
        ? setConfig({type: CONFIG.DEL_FILTER, payload: {id: config.id, filterKeys: [as]}})
        : setConfig({
            type: CONFIG.ADD_FILTER,
            payload: {...cloneObj({[as]: targetFilter}), id},
          });
    } else {
      targetFilter.expr.set.push(filterValue);
      setConfig({
        type: CONFIG.ADD_FILTER,
        payload: {...cloneObj({[as]: targetFilter}), id},
      });
    }
  };
  switch (getShowType(config)) {
    case 'byGradientColor':
      const gap = (ruler.max - ruler.min) / (value.length - 1);
      const formatter = formatterGetter(measureGetter(config, 'color')!, nls);
      const target = dimensionGetter(config, 'color') || measureGetter(config, 'color');
      let label: string = '';
      if (target) {
        label = `${nls[`label_widgetEditor_expression_${target.expression}`] || ''} ${
          (target as Measure).isRecords
            ? nls.label_widgetEditor_recordOpt_label_measure
            : target.label
        }`;
      }
      const items = value.map((item: any, index: number) => {
        const val = formatter(ruler.min + index * gap);
        return {color: item, val};
      });
      const len = items.length;
      return (
        <div className={clsx('gradient-ruler', classes.root)}>
          <div className={classes.ruler} onMouseEnter={onRulerEnter} onMouseLeave={onRulerLeave}>
            {items.map((item: any, index: number) => (
              <div
                key={item.color}
                className={classes.rulerItemContainer}
                data-val={item.val}
                data-color={item.color}
                onMouseEnter={onRulerItemEnter}
                onMouseLeave={onRulerItemLeave}
              >
                <div
                  ref={index === 0 ? rulerMin : index === len - 1 ? rulerMax : null}
                  className={clsx(classes.rulerItem, 'color')}
                  style={{
                    backgroundColor: `${item.color}`,
                  }}
                />
                <div
                  className={clsx(
                    'text',
                    index === len - 1 || index === 0 ? '' : 'ruler-minmax-text-hide'
                  )}
                >
                  {item.val}
                </div>
              </div>
            ))}
          </div>
          <div>{label}</div>
        </div>
      );
    case 'byDistinctColor':
      const colorMeasure = measureGetter(config, 'color')!;
      if (!colorMeasure) {
        return <></>;
      }
      const {as} = colorMeasure;
      return (
        <div className={classes.distinctRoot}>
          {colorItems.map((item: any, index: number) => {
            const {value, color} = item;
            let isSelected = false;
            if (!config.filter[as]) {
              isSelected = true;
            } else {
              isSelected = config.filter[as].expr.set.some((val: string) => val === value);
            }
            const fillColor = isSelected ? color || calDefaultColor(value) : UNSELECTED_COLOR;
            return (
              <div
                key={index}
                className={clsx(classes.colorItemContainer, classes.hover)}
                onClick={() => {
                  upDateFilter(value);
                }}
              >
                <div className={classes.colorItem} style={{backgroundColor: fillColor}} />
                <div>{value}</div>
              </div>
            );
          })}
        </div>
      );
    default:
      return <></>;
  }
};

export default GradientRuler;
