import React, {useState, useEffect, useRef, useContext} from 'react';
import clsx from 'clsx';
import {I18nContext} from '../../contexts/I18nContext';
import {queryContext} from '../../contexts/QueryContext';
import {makeStyles, useTheme} from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import Clear from '@material-ui/icons/Clear';
import {solidOpts, color} from '../../utils/Colors';
import {genEffectClickOutside} from '../../utils/EditorHelper';
import {dimensionGetter, measureGetter} from '../../utils/WidgetHelpers';
import {toSQL} from '../../core/parser/reducer';
import {
  titleMarginBottom,
  subTitleMarginBottom,
  contentMarginBottom,
  genBasicStyle,
} from '../../utils/Theme';
import {QuerySelector as Selector} from '../common/selectors';
import {CONFIG} from '../../utils/Consts';
import {cloneObj} from '../../utils/Helpers';
import {prefixFilter} from '../../utils/Configs';
import {addTextSelfFilter} from '../../widgets/Utils/filters/common';
import {sliceText} from '../../widgets/Utils/Decorators';
const useStyles = makeStyles(theme => ({
  ...genBasicStyle(theme.palette.primary.main),
  title: {
    marginBottom: titleMarginBottom,
    textTransform: 'uppercase',
  },
  subTitle: {
    marginBottom: subTitleMarginBottom,
  },
  label: {
    marginBottom: '5px',
  },
  card: {
    padding: theme.spacing(1),
  },
  colorRooter: {
    border: 'solid',
    borderRadius: '5px',
    borderColor: theme.palette.grey[700],
    borderWidth: '.5px',
    padding: '0 5px',
    marginBottom: '2px',
  },
  colorContainer: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '5px 5px',
    marginBottom: contentMarginBottom,
  },
  colorItem: {
    width: '24px',
    height: '24px',
    marginRight: '5px',
    cursor: 'pointer',
  },
  currColorItem: {
    width: '24px',
    height: '24px',
    marginRight: '10px',
    cursor: 'pointer',
  },
  colName: {
    flexGrow: 1,
    display: 'flex',
    alignItems: 'center',
    height: '24px',
  },
  optsRoot: {
    marginBottom: '5px',
  },
  optContainer: {
    display: 'flex',
    padding: '5px',
    cursor: 'pointer',
  },
  selected: {
    border: 'solid',
    borderWidth: '3px',
    borderColor: theme.palette.grey[50],
  },
}));

const VisualDataMapping = (props: any) => {
  const {nls} = useContext(I18nContext);
  const {getTxtDistinctVal} = useContext(queryContext);
  const theme = useTheme();
  const classes = useStyles(theme);
  const {config, setConfig} = props;
  const {source, colorItems = []} = config;
  const colorTarget = dimensionGetter(config, 'color') || measureGetter(config, 'color');
  const colorType = colorTarget ? 'byDistinctColor' : 'byMeasures';
  const colorList = colorItems.map((item: any) => {
    const copyItem = cloneObj(item);
    if (copyItem.isRecords) {
      copyItem.label = nls.label_widgetEditor_recordOpt_label_measure;
    }
    copyItem.label = sliceText(copyItem.label);
    copyItem.color = copyItem.color || color(copyItem.color);
    return copyItem;
  });
  const ref = useRef(null);
  const [selectedOpt, setSelectedOpt] = useState();

  const changeColor = (e: any, as: string) => {
    const target = colorItems.find((item: any) => item.as === as);
    target.color = e.target.dataset.color;
    setConfig({type: CONFIG.ADD_COLORITEMS, payload: [target]});
  };

  const genSolidOpts = ({opts, onClick, as, color}: any) => {
    return (
      <div className={classes.optContainer}>
        {opts.map((opt: any) => {
          const isSelected = opt.value === color;
          return (
            <div
              className={clsx(classes.colorItem, isSelected ? classes.selected : '')}
              onClick={(e: any) => onClick(e, as)}
              key={opt.key}
              style={{
                backgroundColor: opt.value,
              }}
              data-color={opt.value}
            />
          );
        })}
      </div>
    );
  };

  const addColorItem = (val: any) => {
    const colorItem = JSON.parse(val);
    if (!colorItems.find((s: any) => s.as === colorItem.as)) {
      addTextSelfFilter({
        val: colorItem.as,
        key: colorTarget!.as,
        config,
        setConfig,
      });
      setConfig({payload: [colorItem], type: CONFIG.ADD_COLORITEMS});
    }
  };

  const deleteColorItem = (as: any) => {
    // keep at least one colorItem
    if (colorItems.length === 1) {
      return;
    }

    const colorItem = colorItems.find((item: any) => item.as === as);
    setConfig({
      type: CONFIG.DEL_COLORITEMS,
      payload: [colorItem],
    });
    if (colorTarget) {
      const targetKey = prefixFilter('selfFilter', colorTarget.as);
      const targetSelfFilter = config.selfFilter[targetKey];
      targetSelfFilter.expr.set = targetSelfFilter.expr.set.filter(
        (val: string) => val !== colorItem.as
      );
      setConfig({type: CONFIG.ADD_SELF_FILTER, payload: {[targetKey]: targetSelfFilter}});
    }
  };

  const _parseRes = (res: any) => {
    return res.map((r: any) => {
      const colName = Object.keys(r)[0];
      const value = r[colName];
      return {
        label: value,
        value: JSON.stringify({
          colName,
          as: value,
          label: value,
          value,
          color: color(value),
        }),
      };
    });
  };
  const query = async (typingText: string | undefined, limit: number) => {
    const existOpts = colorItems.map((item: any) => `'${item.as}'`).join(', ');
    const NotIn = existOpts ? `${colorTarget!.value} NOT IN (${existOpts})` : '';
    const Like = typingText ? `${colorTarget!.value} LIKE '%${typingText}%'` : '';
    const sql = toSQL({
      select: [`DISTINCT ${colorTarget!.value}`],
      from: source,
      where: [NotIn, Like].filter((str: string) => !!str),
      limit: limit,
    });
    const res = await getTxtDistinctVal(sql);
    return _parseRes(res || []);
  };

  // hide options when click outSide
  useEffect(() => {
    const hideOpts = genEffectClickOutside(ref.current, setSelectedOpt, '');
    document.body.addEventListener('click', hideOpts);
    if (!config.colorItems) {
      setConfig({payload: colorItems, type: CONFIG.ADD_COLORITEMS});
    }
    return () => {
      document.body.removeEventListener('click', hideOpts);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div ref={ref}>
      <div className={classes.title}>{nls.label_visual_data_mapping}</div>
      <div className={classes.subTitle}>{nls.label_custom_color}</div>
      <Card classes={{root: classes.card}}>
        {colorList.map((item: any, index: number) => {
          const {label, as, color} = item;
          return (
            <div className={classes.colorRooter} key={index}>
              <div className={classes.colorContainer} key={index}>
                <div
                  className={classes.currColorItem}
                  style={{backgroundColor: item.color}}
                  onClick={() => setSelectedOpt(as)}
                />
                <div className={classes.colName}>{label}</div>
                {colorType === 'byDistinctColor' && (
                  <Clear classes={{root: classes.hover}} onClick={() => deleteColorItem(as)} />
                )}
              </div>
              {selectedOpt === as &&
                genSolidOpts({
                  opts: solidOpts,
                  onClick: changeColor,
                  as: as,
                  color,
                })}
            </div>
          );
        })}
        {colorType === 'byDistinctColor' && (
          <Selector
            currOpt={{}}
            placeholder={nls.label_widgetEditorDisplay_VisualDataMapping_placeholder}
            query={query}
            onOptionChange={addColorItem}
          />
        )}
      </Card>
    </div>
  );
};

export default VisualDataMapping;
