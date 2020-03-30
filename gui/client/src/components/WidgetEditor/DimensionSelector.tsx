import React, {FC, useState, useEffect, useRef, useContext} from 'react';
import {makeStyles, useTheme} from '@material-ui/core/styles';
import clsx from 'clsx';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import Input from '@material-ui/core/Input';
import AccessTime from '@material-ui/icons/AccessTime';
import Clear from '@material-ui/icons/Clear';
import {rootContext} from '../../contexts/RootContext';
import {queryContext} from '../../contexts/QueryContext';
import {I18nContext} from '../../contexts/I18nContext';
import {DimensionSelectorProps, Dimension} from '../../types';
import {TimeBin, timeBinMap} from '../../utils/Time';
import {
  calStatus,
  genRangeQuery,
  filterColumns,
  genEffectClickOutside,
} from '../../utils/EditorHelper';
import {COLUMN_TYPE} from '../../utils/Consts';
import {getValidTimeBinOpts} from '../../utils/Binning';
import {cloneObj, id as genID} from '../../utils/Helpers';
import {isDateCol, isTextCol, isNumCol} from '../../utils/ColTypes';
import {Status} from '../../types/Editor';
import Bin from './Bin';
import NoOtherOpt from '../common/NoMoreOpt';
import CustomSqlInput from './CustomSqlInput';
import CustomSqlOpt from './CustomSqlOpt';
import genDimensionSelectorStyles from './DimensionSelector.style';

type NewColumn = {
  col_name: string;
  type: string;
  label?: string;
  short?: string;
  key?: string;
  isCustom?: boolean;
};

const useStyles = makeStyles(theme => genDimensionSelectorStyles(theme) as any) as Function;

const DimensionSelector: FC<DimensionSelectorProps> = (props: DimensionSelectorProps) => {
  const {getData} = useContext(queryContext);
  const {nls} = useContext(I18nContext);
  const {setDialog} = useContext(rootContext);
  const theme = useTheme();
  const classes = useStyles(theme);
  const {
    id,
    source,
    setting,
    placeholder,
    dimension,
    options,
    deleteDimension,
    addDimension = () => {},
    dLength = 0,
    enableAddColor = true,
  } = props;
  const {short = '', key, isNotUseBin, onAdd, onDelete, expression} = setting;
  const binTypeOrigin = dimension && _getBinType(dimension);
  const binType = binTypeOrigin ? nls[`label_widgetEditor_${binTypeOrigin}`] : '';
  // status: selected, selectColumn, selectBin, add
  const [status, setStatus] = useState(Status.ADD);
  const [typingText, setTypingText] = useState();

  const filteredOpts = filterColumns(typingText, options);
  const rootDom = useRef(null);
  const isUseInput: any = useRef(null);
  const _showReqireLabel = nls[`label_widgetEditor_requireLabel_${short}`] || short;
  // use for AddSelector in any | requiedOneAtLeast situation
  useEffect(() => {
    setStatus(dimension ? Status.SELECTED : Status.ADD);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dLength]);

  const isFirstRun = useRef(true);
  useEffect(() => {
    if (isFirstRun.current) {
      isFirstRun.current = false;
      return;
    }
    const status = dimension ? calStatus(dimension!, setting, 'dimension') : Status.ADD;
    setStatus(status);
    // add dimensionLabel for CustomInput change, do not delete it
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dimension && dimension.value + dimension.label]);

  // hide opts when click outside
  useEffect(() => {
    const hideDimension = genEffectClickOutside(
      rootDom.current,
      setStatus,
      dimension && dimension.value ? Status.SELECTED : Status.ADD
    );
    document.body.addEventListener('click', hideDimension);
    return () => {
      document.body.removeEventListener('click', hideDimension);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [status]);

  const editDimension = () => {
    const isNoDimension = !dimension;
    const noPopUp = isNoDimension || isNotUseBin || (dimension && dimension.type === 'text');
    setStatus(noPopUp ? Status.SELECT_COLUMN : Status.SELECT_BIN);
  };

  const afterColumnSelected = () => setTypingText(undefined);

  const onCustomSqlOptClick = () => {
    setStatus(Status.CUSTOM);
  };

  const onReceiveInvalidBinRange = () => {
    setDialog({
      open: true,
      title: nls.label_widgetEditor_invalid_binRange_title,
      content: nls.label_widgetEditor_invalid_binRange_content,
      onConfirm: () => {
        setStatus(dimension ? Status.SELECTED : Status.ADD);
      },
    });
  };

  const onColumnSelect = async (e: any) => {
    const col_name = e.target.textContent;
    const type = e.target.dataset.type;
    if (dimension && dimension.value === col_name) {
      setStatus(Status.SELECTED);
      afterColumnSelected();
      return;
    }
    const as = dimension && dimension.as;
    const newDimension: any = _genNewDimension(
      {
        col_name,
        key,
        type,
        short,
      },
      as,
      expression
    );
    if (isTextCol(type)) {
      addDimension(newDimension, onAdd);
      return;
    }
    if (isNotUseBin) {
      addDimension(
        {
          ...newDimension,
          isBinned: false,
          isNotUseBin: true,
        },
        onAdd
      );
      return;
    }
    const params = genRangeQuery(col_name, source);
    const res = await getData(params);
    const {minimum, maximum} = res[0];
    if (minimum === null || minimum === undefined || maximum === null || maximum === undefined) {
      onReceiveInvalidBinRange();
      return;
    }
    addDimension(_addBinRange(newDimension, res[0]), onAdd);
    afterColumnSelected();
  };

  const addCustomDimension = async (customColName: string, expression: string) => {
    if (dimension && customColName === dimension.label && expression === dimension.value) {
      setStatus(Status.SELECTED);
      return;
    }
    if (dimension && expression === dimension.value && dimension.isCustom) {
      const newDimension = cloneObj(dimension);
      newDimension.label = customColName;
      addDimension(newDimension);
      return;
    }
    const as = dimension && dimension.as;
    const newDimension = _genNewDimension(
      {
        key,
        type: 'float',
        label: customColName,
        col_name: expression,
        short,
        isCustom: true,
      },
      as
    );
    if (isNotUseBin) {
      addDimension({...newDimension, isBinned: false}, onAdd);
    } else {
      const params = genRangeQuery(expression, source);
      const res = await getData(params);
      const {minimum, maximum} = res[0];
      if (typeof minimum !== 'number' || typeof maximum !== 'number') {
        onReceiveInvalidBinRange();
        return;
      }
      addDimension(_addBinRange(newDimension, res[0]), onAdd);
    }
  };

  const onBlur = () => {
    if (status !== Status.SELECT_BIN) {
      setStatus(dimension ? Status.SELECTED : Status.ADD);
    }
  };

  const onInputClick = () => {
    setStatus(
      dimension && dimension.isCustom && !isUseInput.current ? Status.CUSTOM : Status.SELECT_COLUMN
    );
    isUseInput.current = false;
  };

  const onInputChange = (e: any) => {
    setTypingText(e.target.value || '');
  };

  const onCustomInputCancle = () => {
    isUseInput.current = true;
    setStatus(Status.SELECT_COLUMN);
  };

  const defaultPlaceholder = {name: '', funcText: ''};
  return (
    <div ref={rootDom}>
      {status === Status.SELECTED && (
        <div className={classes.root}>
          <div className={clsx(classes.button, short ? classes.short : classes.hidden)}>
            {_showReqireLabel}
          </div>
          <div className={clsx(classes.content, classes.hover)} onClick={editDimension}>
            {`${binType} ${dimension && dimension.label}`}
          </div>
          {deleteDimension && (
            <Clear
              classes={{
                root: classes.hover,
              }}
              className={clsx(classes.button, classes.deleteSeletor)}
              onClick={() => {
                deleteDimension(dimension!.as, onDelete);
              }}
            />
          )}
        </div>
      )}
      {(status === Status.SELECT_COLUMN || status === Status.SELECT_BIN) && (
        <>
          <Input
            className={classes.input}
            fullWidth={true}
            value={typingText === undefined ? (dimension && dimension.label) || '' : typingText}
            onChange={onInputChange}
            onBlur={onBlur}
            onClick={onInputClick}
          />
        </>
      )}
      {status === Status.SELECT_COLUMN && (
        <List component="ul" className={classes.options}>
          <CustomSqlOpt
            placeholder={nls.label_widgetEditor_customOpt_placeholder_dimension}
            onClick={onCustomSqlOptClick}
          />
          {filteredOpts.map((option: any, index: number) => {
            const [isNum, isText, isDate] = [
              isNumCol(option.data_type),
              isTextCol(option.data_type),
              isDateCol(option.data_type),
            ];
            return (
              <ListItem
                classes={{
                  gutters: classes.customGutters,
                }}
                key={`${option.col_name} ${index}`}
                button
                divider
              >
                <ListItemText
                  classes={{
                    root: classes.customTextRoot,
                  }}
                  primary={
                    <span className={classes.option} onMouseDown={onColumnSelect}>
                      <span className={classes.optionLabel} data-type={option.data_type}>
                        {option.col_name}
                      </span>
                      {isDate && (
                        <AccessTime
                          classes={{
                            root: classes.icon,
                          }}
                          fontSize="small"
                        />
                      )}
                      {isText && <span className={classes.icon}>a</span>}
                      {isNum && <span className={classes.icon}>#</span>}
                    </span>
                  }
                />
              </ListItem>
            );
          })}
          {filteredOpts.length === 0 && (
            <NoOtherOpt
              onClick={() => {
                setStatus(Status.ADD);
              }}
            />
          )}
        </List>
      )}
      {status === Status.SELECT_BIN && (
        <Bin
          id={id}
          source={source}
          dimension={dimension!}
          staticRange={dimension!.staticRange}
          addDimension={addDimension}
          onAdd={onAdd}
        />
      )}
      {status === Status.CUSTOM && (
        <CustomSqlInput
          onCancle={onCustomInputCancle}
          onSave={addCustomDimension}
          currVal={{label: dimension && dimension.label, value: dimension && dimension.value}}
          placeholder={defaultPlaceholder}
        />
      )}
      {status === Status.ADD && (
        <div className={clsx(classes.root, enableAddColor ? classes.hover : classes.disable)}>
          <div
            className={clsx(
              classes.button,
              short ? classes.short : classes.hidden,
              enableAddColor ? '' : classes.disableReq
            )}
          >
            {_showReqireLabel}
          </div>
          <div
            className={`${classes.addDimension}`}
            onClick={enableAddColor ? editDimension : undefined}
          >
            {placeholder}
          </div>
        </div>
      )}
    </div>
  );
};

// helpers
const _genNewDimension = (
  newColumn: NewColumn,
  originAs?: string,
  expression: string | undefined = undefined
): Dimension => {
  const {key, col_name, type, isCustom = false, short, label} = newColumn;
  let _binType: string = '';
  if (isNumCol(type)) {
    _binType = 'numBin';
  }
  if (isDateCol(type)) {
    _binType = 'dateBin';
  }
  if (isTextCol(type)) {
    _binType = 'textBin';
  }

  let newDimension: Dimension = {
    label: label || (col_name as string),
    short,
    value: col_name as string,
    type,
    format: 'auto',
    as: originAs || key || (genID(col_name) as string),
  };
  switch (_binType) {
    case 'numBin':
      newDimension = {
        ...newDimension,
        isBinned: true,
        maxbins: 12,
      };
      break;
    case 'dateBin':
      newDimension = {
        ...newDimension,
        isBinned: true,
        extract: false,
      };
      break;
    case 'textBin':
    default:
      newDimension = {
        ...newDimension,
        type: COLUMN_TYPE.TEXT,
      };
      break;
  }
  if (isCustom) {
    newDimension.isCustom = true;
  }
  if (expression) {
    newDimension.expression = expression;
  }
  return newDimension;
};

const _getBinType = (dimension: Dimension) => {
  const {type, isNotUseBin, isBinned, extract = false} = dimension;
  const isNotBined = isNotUseBin || !isBinned || !type;
  const binType = extract ? 'extract' : 'bin';
  return isNotBined ? '' : binType;
};
const _addBinRange = (
  newDimension: Dimension,
  res: {minimum: string | number; maximum: string | number}
): Dimension => {
  const {minimum, maximum} = res;
  newDimension.min = minimum;
  newDimension.max = maximum;
  newDimension.extent = [minimum, maximum];
  newDimension.staticRange = [minimum, maximum];
  if (isDateCol(newDimension.type)) {
    const validTimeBinOpts = getValidTimeBinOpts(minimum, maximum, false);
    newDimension.timeBin = validTimeBinOpts[Math.floor(validTimeBinOpts.length / 2)]!
      .value as TimeBin;
    newDimension.binningResolution = timeBinMap[newDimension.timeBin]!;
  }
  return newDimension;
};
export default DimensionSelector;
