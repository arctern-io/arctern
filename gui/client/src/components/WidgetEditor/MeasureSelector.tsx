import React, {FC, useState, useEffect, useRef, useContext} from 'react';
import {makeStyles, useTheme} from '@material-ui/core/styles';
import clsx from 'clsx';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import AccessTime from '@material-ui/icons/AccessTime';
import Input from '@material-ui/core/Input';
import Clear from '@material-ui/icons/Clear';
import {I18nContext} from '../../contexts/I18nContext';
import {MeasureSelectorProps, Measure} from '../../types';
import {Status} from '../../types/Editor';
import ExpressionDropdown from './ExpressionDropdown';
import NoOtherOpt from '../common/NoMoreOpt';
import CustomSqlOpt from './CustomSqlOpt';
import RecordSqlOpt from './RecordSqlOpt';
import CustomSqlInput from './CustomSqlInput';
import {isNumCol, isDateCol, isTextCol, getColType} from '../../utils/ColTypes';
import {
  calStatus,
  filterColumns,
  genEffectClickOutside,
  settingUsePopUp,
  measureUsePopUp,
  Column,
} from '../../utils/EditorHelper';
import {id as genID} from '../../utils/Helpers';
import genMeasureSelectorStyles from './MeasureSelector.style';

const useStyles = makeStyles(theme => genMeasureSelectorStyles(theme) as any) as Function;

type NewColumn = {
  col_name: string;
  type: string;
  label?: string;
  key?: string;
  isCustom?: boolean;
  isRecords?: boolean;
};

const MeasureSelector: FC<MeasureSelectorProps> = props => {
  const theme = useTheme();
  const classes = useStyles(theme);
  const {nls} = useContext(I18nContext);
  const {
    setting,
    placeholder = 'Add Measure',
    measure,
    options = [],
    addMeasure,
    deleteMeasure,
    mLength = 0,
    isEnableAddMore = true,
  } = props;
  const measureLabel = (measure && measure.label) || '';
  const {short = '', labelPlus = '', key, expressions, onAdd, onDelete} = setting;
  const [status, setStatus] = useState(Status.ADD);
  const [typingText, setTypingText]: any = useState();

  const filteredOpts = filterColumns(typingText, options);
  const rootRef = useRef(null);
  const _showReqireLabel = nls[`label_widgetEditor_requireLabel_${short}`] || short;
  const _showExpression =
    nls[`label_widgetEditor_expression_${measure && measure.expression}`] || '';
  const _showLabel =
    measure && measure.isRecords
      ? nls.label_widgetEditor_recordOpt_label_measure
      : measure && measure.label;
  // hide measureSelector when click outsite
  useEffect(() => {
    const hideMeasure = genEffectClickOutside(
      rootRef.current,
      setStatus,
      measure ? Status.SELECTED : Status.ADD
    );
    document.body.addEventListener('click', hideMeasure);
    return () => {
      document.body.removeEventListener('click', hideMeasure);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [measure]);

  const isFirstRun = useRef(true);
  useEffect(() => {
    if (isFirstRun.current) {
      setStatus(measure ? Status.SELECTED : Status.ADD);
      isFirstRun.current = false;
      return;
    }
    if (!measure) {
      setStatus(Status.ADD);
      return;
    }
    if (measure && measure.isCustom) {
      setStatus(Status.SELECTED);
      return;
    }
    const _status = calStatus(measure!, setting, 'measure');
    setStatus(_status);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [measure && measure.value]);

  // For making AddButton Status.ADD
  useEffect(() => {
    setStatus(measure ? Status.SELECTED : Status.ADD);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mLength]);

  // click option
  const _selectColomn = (e: any) => {
    const col_name = e.target.dataset.value;
    if (measure && measure.value === col_name) {
      setStatus(Status.SELECTED);
      return;
    } else {
      const type = e.target.dataset.type;
      const as = (measure && measure.as) || '';
      const newMeasure = _genNewMeasure({key, col_name, type}, as, expressions);
      addMeasure(newMeasure, onAdd);
    }
  };

  const editMeasure = () => {
    if (measure && measure.isCustom) {
      setStatus(Status.CUSTOM);
      return;
    }
    if (measureUsePopUp(measure!) && settingUsePopUp(expressions)) {
      setStatus(Status.SELECT_EXPRESSION);
      return;
    }
    setStatus(Status.SELECT_COLUMN);
  };

  const setSelectColumn = (e: any) => {
    setTypingText();
    setStatus(Status.SELECT_COLUMN);
  };

  const onInputChange = (e: any) => {
    setTypingText(e.target.value || '');
  };

  const _addCustomOpt = () => {
    setStatus(Status.CUSTOM);
  };

  const _addRecordOpt = (e: any) => {
    if (measure && measure.isRecords) {
      setStatus(Status.SELECTED);
      return;
    } else {
      const newMeasure: any = _genNewMeasure(
        {
          isRecords: true,
          type: 'int',
          key,
          label: 'Records',
          col_name: 'Records',
        },
        (measure && measure.as) || '',
        ['count']
      );
      addMeasure(newMeasure, onAdd);
    }
  };

  const _saveCustomMeasure = (customColname: string, expression: string) => {
    // as must be without space and start with no number type
    // type only support number at the moment, add other types later
    const newMeasure = _genNewMeasure(
      {
        key,
        label: customColname,
        col_name: expression,
        type: 'float8',
        isCustom: true,
      },
      (measure && measure.as) || '',
      ['project']
    );
    addMeasure(newMeasure, onAdd);
  };

  const defaultPlaceholder = {name: '', funcText: ''};
  return (
    <div ref={rootRef}>
      {status === Status.SELECTED && (
        <div className={classes.root}>
          <div className={clsx(classes.button, short ? classes.short : classes.hidden)}>
            {_showReqireLabel.concat(labelPlus)}
          </div>
          <div className={`${classes.content} ${classes.hover}`} onClick={editMeasure}>
            {`${_showExpression} ${_showLabel}`}
          </div>
          {deleteMeasure && (
            <Clear
              classes={{root: classes.hover}}
              className={clsx(classes.button, classes.deleteSeletor)}
              onClick={() => {
                deleteMeasure(measure!.as, onDelete);
              }}
            />
          )}
        </div>
      )}
      {(status === Status.SELECT_COLUMN || status === Status.SELECT_EXPRESSION) && (
        <Input
          className={classes.input}
          fullWidth={true}
          value={typingText !== undefined ? typingText : (measure && measure.value) || ''}
          onChange={onInputChange}
          onClick={setSelectColumn}
        />
      )}
      {status === Status.SELECT_COLUMN && (
        <List component="ul" dense={true} className={classes.options}>
          <CustomSqlOpt
            placeholder={nls.label_widgetEditor_customOpt_placeholder_measure}
            onClick={_addCustomOpt}
          />
          <RecordSqlOpt
            placeholder={nls.label_widgetEditor_recordOpt_placeholder_measure}
            onClick={_addRecordOpt}
          />
          {filteredOpts.map((option: Column, index: number) => {
            const [isNum, isText, isDate] = [
              isNumCol(option.data_type),
              isTextCol(option.data_type),
              isDateCol(option.data_type),
            ];
            return (
              <ListItem
                classes={{gutters: classes.customGutters}}
                key={`${option.col_name} ${index}`}
                button
                divider
              >
                <ListItemText
                  classes={{root: classes.customTextRoot}}
                  primary={
                    <span className={classes.option} onMouseDown={_selectColomn}>
                      <span
                        className={classes.optionLabel}
                        data-type={option.data_type}
                        data-value={option.col_name}
                      >
                        {option.col_name}
                      </span>
                      {isDate && (
                        <AccessTime fontSize="small" classes={{root: classes.customIcon}} />
                      )}
                      {isText && <span className={classes.customIcon}>a</span>}
                      {isNum && <span className={classes.customIcon}>#</span>}
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
      {status === Status.SELECT_EXPRESSION && (
        <ExpressionDropdown expressions={expressions} measure={measure!} addMeasure={addMeasure} />
      )}
      {status === Status.CUSTOM && (
        <CustomSqlInput
          onCancel={() => {
            setStatus(Status.SELECT_COLUMN);
          }}
          onSave={_saveCustomMeasure}
          currVal={{measureLabel, value: measure && measure.value}}
          placeholder={defaultPlaceholder}
        />
      )}
      {status === Status.ADD && (
        <div className={clsx(classes.root, isEnableAddMore ? classes.hover : classes.disable)}>
          <div
            className={clsx(
              classes.button,
              short ? classes.short : classes.hidden,
              isEnableAddMore ? '' : classes.disableReq
            )}
          >
            {_showReqireLabel}
          </div>
          <div className={classes.addMeasure} onClick={isEnableAddMore ? editMeasure : undefined}>
            {placeholder}
          </div>
        </div>
      )}
    </div>
  );
};
const _genNewMeasure = (newSelected: NewColumn, originAs: string, expressions: string[]) => {
  const {key, col_name, label, type, isCustom = false, isRecords = false} = newSelected;
  const newMeasure: Measure = {
    type,
    value: col_name,
    label: label || col_name,
    as: originAs || key || genID('measure'),
    format: 'auto',
    isCustom,
    isRecords,
    expression: '',
  };
  const isProjectExpression =
    isCustom || (expressions[0] === 'project' && expressions.length === 1);
  if (isProjectExpression) {
    return {...newMeasure, expression: 'project'};
  }
  if (isRecords) {
    return {
      ...newMeasure,
      expression: 'count',
      value: '*',
      as: originAs || key || 'countval',
    };
  }
  if (expressions.length === 1) {
    return {
      ...newMeasure,
      expression: expressions[0],
    };
  }
  switch (getColType(type)) {
    case 'text':
      newMeasure.expression = 'unique';
      break;
    case 'number':
    default:
      newMeasure.expression = 'avg';
      break;
  }
  return newMeasure;
};
export default MeasureSelector;
