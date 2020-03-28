import React, { useState, useEffect, useRef } from 'react';
import { useStatus } from './Helper'
import { makeStyles } from '@material-ui/core/styles';
import clsx from 'clsx';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import Input from '@material-ui/core/Input';
import Clear from '@material-ui/icons/Clear';
import { QueryCount, genEffectClickOutside } from '../../../utils/EditorHelper';
import NoOtherOpt from '../NoMoreOpt';
import ShowMore from '../ShowMore';
import genSelectorStyle from './Selector.style';

const _getValidOpts = (currOpt: any, opts: any[], useCurrOpt: boolean) => {
  return useCurrOpt ? opts : opts.filter((opt: any) => opt.value !== currOpt.value);
};

const _genStatus = (visible_options_length: number, valid_options: any[]) => {
  return {
    useShowMore: visible_options_length < valid_options.length,
    useNoMoreOpt: visible_options_length === valid_options.length,
  };
};
export const filterOptions = (filter_text: string = '', valid_options: any[]) => {
  const regex = new RegExp(filter_text, 'ig');
  return filter_text ? valid_options.filter((item: any) => regex.test(item.value)) : valid_options;
};

// state hooks
export function useOptions(valid_options: any[] = []) {
  const [filter_text, setFilterText] = useState();
  const [filtered_options, setFilteredOptions] = useState(valid_options)
  const [visible_options, setVisibleOptions]: any = useState(filtered_options.slice(0, QueryCount));

  const isFirstRun = useRef(true);
  const isFirstRun_1 = useRef(true);
  useEffect(() => {
    if (isFirstRun.current) {
      isFirstRun.current = false;
      return;
    }
    const new_filtered_options = filterOptions(filter_text, valid_options);
    setFilteredOptions(new_filtered_options);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter_text])

  useEffect(() => {
    if (isFirstRun_1.current) {
      isFirstRun_1.current = false;
      return;
    }
    const new_visible_options = filtered_options.slice(0, QueryCount);
    setVisibleOptions(new_visible_options);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(filtered_options)])

  const changeFilter = (e: any) => {
    const new_filter_text = e.target.value.trim() || "";
    setFilterText(new_filter_text);
  }
  const clearFilter = () => {
    setFilterText(undefined);
    setFilteredOptions(valid_options);
  }
  const showMore = () => {
    const new_length = visible_options.length + QueryCount;
    const new_visible_options = filtered_options.slice(0, new_length);
    setVisibleOptions(new_visible_options)
  }
  return {
    filter_text, filtered_options, visible_options,
    clearFilter, changeFilter, showMore
  }
}
const SimpleSelector = (props: any) => {
  const classes = makeStyles(theme => genSelectorStyle(theme) as any)() as any;
  const {
    currOpt = { value: '' },
    options = [],
    placeholder = '',
    onOptionChange,
    onMouseOver = () => { },
    onDelete,
    isShowCurrOpt = false,
  } = props;
  const { value, label, tip = '' } = currOpt;
  const validOptions = _getValidOpts(currOpt, options, isShowCurrOpt);

  const { status, determineStatus, setSelectingStatus } = useStatus(value ? 'selected' : 'add')
  const { filter_text, filtered_options, visible_options, clearFilter, changeFilter, showMore } = useOptions(validOptions)

  const rootDom = useRef(null);
  const { useShowMore, useNoMoreOpt } = _genStatus(visible_options.length, filtered_options);

  useEffect(() => {
    determineStatus(currOpt.value);
    const hide = genEffectClickOutside(
      rootDom.current,
      determineStatus,
      currOpt.value
    );
    document.body.addEventListener('click', hide);
    return () => {
      document.body.removeEventListener('click', hide);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify([...options, currOpt])]);

  // click option
  const onSelectColumn = (e: any) => {
    const _value = e.currentTarget.dataset.value;
    clearFilter();
    value === _value
      ? determineStatus(_value)
      : onOptionChange(_value);
  };

  const toSelectColName = async () => {
    setSelectingStatus();
  };

  const _onMouseOver = (e: any) => {
    const val = e.currentTarget.dataset.value;
    onMouseOver(val);
  };

  return (
    <div ref={rootDom}>
      {status === 'selected' && (
        <div className={clsx(classes.root, !onDelete && classes.hover)}>
          <div
            className={clsx(classes.buttonStatus, onDelete && classes.hover)}
            onClick={toSelectColName}
          >
            {!onDelete && (
              <div className={clsx(tip ? classes.content : classes.onlyContent, classes.hover)}>
                <div>{label}</div>
                <div className={classes.tip}>{tip}</div>
              </div>
            )}
            {onDelete && (
              <>
                <div className={clsx(classes.content, classes.hover)}>{label}</div>
                {label && (
                  <div className={clsx(classes.clear, classes.hover)}>
                    <Clear
                      onClick={() => {
                        onDelete(label);
                      }}
                    />
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      )}
      {status === 'selectColumn' && (
        <>
          <Input
            className={classes.input}
            fullWidth
            type="text"
            value={filter_text !== undefined ? filter_text : label || value || ''}
            onChange={changeFilter}
          />
          <List component="ul" className={classes.options}>
            {visible_options.map((option: any, index: number) => {
              return (
                <ListItem
                  classes={{ gutters: classes.customGutters }}
                  key={`${option.colName} ${index}`}
                  button
                  divider
                >
                  <ListItemText
                    classes={{ root: classes.customTextRoot }}
                    primary={
                      <span
                        className={classes.option}
                        data-type={option.type}
                        data-value={option.value}
                        onMouseDown={onSelectColumn}
                        onMouseOver={_onMouseOver}
                      >
                        <span className={classes.optionLabel}>{option.label}</span>
                        <span className={classes.customIcon}>{option.tip || ''}</span>
                      </span>
                    }
                  />
                </ListItem>
              );
            })}
            {useShowMore && <ShowMore onClick={() => showMore()} />}
            {useNoMoreOpt && <NoOtherOpt />}
          </List>
        </>
      )}
      {status === 'add' && (
        <div className={clsx(classes.root, classes.hover)}>
          <div className={clsx(classes.hover, classes.onlyContent)} onClick={toSelectColName}>
            {placeholder}
          </div>
        </div>
      )}
    </div>
  );
};

export default SimpleSelector;
