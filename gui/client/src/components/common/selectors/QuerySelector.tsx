import React, { useState, useEffect, useRef } from 'react';
import { makeStyles } from '@material-ui/core/styles';
import { useStatus } from './Helper'
import clsx from 'clsx';
import List from '@material-ui/core/List';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import Input from '@material-ui/core/Input';
import Clear from '@material-ui/icons/Clear';
import NoOtherOpt from '../NoMoreOpt';
import ShowMore from '../ShowMore';
import { QueryCount, genEffectClickOutside } from '../../../utils/EditorHelper';
import { sliceText } from '../../../widgets/Utils/Decorators';
import genSelectorStyle from './Selector.style';

const _handleOverflowLabel = (res: any[]) => {
  return res.map((item: any) => {
    return {
      ...item,
      label: sliceText(item.label),
    };
  });
};
const _genLastOpt = (visible_options_length: number) => {
  if (visible_options_length < QueryCount) {
    return { useShowMore: false, useNoMoreOpt: true };
  }
  return {
    useShowMore: visible_options_length % QueryCount === 0,
    useNoMoreOpt: visible_options_length % QueryCount > 0,
  };
};
export function useOptions(query: Function) {
  const [visible_options, setVisibleOptions]: any = useState([]);
  const [filter_text, setFilterText] = useState();
  const limitRef: any = useRef(QueryCount);
  const isSubscribed = useRef(true);
  const isFirstRun = useRef(true);
  useEffect(() => {
    if (isFirstRun.current) {
      isFirstRun.current = false;
      return;
    }
    limitRef.current = 0;
    showMore(filter_text)
    return () => {
      isSubscribed.current = false;
    }
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filter_text])

  const showMore = async (filter_text: string | undefined, init_len: number = limitRef.current) => {
    const new_len = init_len + QueryCount;
    const res = await query(filter_text, new_len);
    if (isSubscribed.current && res) {
      limitRef.current = res.length;
      setVisibleOptions(_handleOverflowLabel(res));
    }
  }
  const changeFilter = (e: any) => {
    const value = e.target.value.trim() || ""
    limitRef.current = QueryCount
    setFilterText(value);
  }
  const resetFilter = () => {
    setFilterText(undefined);
  }
  return {
    filter_text, visible_options, changeFilter,
    showMore, resetFilter
  }
}

const QuerySelector = (props: any) => {
  const classes = makeStyles(theme => genSelectorStyle(theme))() as any;
  const {
    currOpt = { value: '' },
    query = () => { },
    placeholder = '',
    onOptionChange,
    onMouseOver = () => { },
    onDelete,
  } = props;
  const { value, label, tip = '' } = currOpt;
  const { status, determineStatus, setSelectingStatus } = useStatus(value ? 'selected' : 'add')
  const { filter_text, visible_options, changeFilter, showMore, resetFilter } = useOptions(query)
  const { useShowMore, useNoMoreOpt } = _genLastOpt(visible_options.length);
  const rootDom = useRef(null);

  useEffect(() => {
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
  }, [JSON.stringify([currOpt])]);

  // click option
  const selectColumn = (e: any) => {
    const _value = e.currentTarget.dataset.value;
    value === _value ? determineStatus('selected') : onOptionChange(_value);
    resetFilter()
  };
  //TODO: could use cache here
  const toSelectColName = async () => {
    await showMore(filter_text, 0);
    setSelectingStatus()
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
                <div>{label || value}</div>
                <div className={classes.tip}>{tip}</div>
              </div>
            )}
            {onDelete && (
              <>
                <div className={clsx(classes.content, classes.hover)}>{label || value}</div>
                {(label || value) && (
                  <div className={clsx(classes.clear, classes.hover)}>
                    <Clear
                      onClick={() => {
                        onDelete(label || value);
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
                        onMouseDown={selectColumn}
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
            {useShowMore && <ShowMore onClick={() => showMore(filter_text)} />}
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

export default QuerySelector;
