import React, {FC, useContext, useEffect} from 'react';
import {SimpleSelector as Selector} from '../common/selectors';
import {makeStyles} from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import {I18nContext} from '../../contexts/I18nContext';
import {isRecordExist} from '../../utils/EditorHelper';
import {CONFIG} from '../../utils/Consts';
import {Dimension, Measure, WidgetConfig} from '../../types';

type Sort = {
  name: string;
  order: 'descending' | 'accending';
};

type SortOption = {
  value: string;
  label: string;
  tip?: string;
};
export const DEFAULT_SORT: Sort = {
  name: '',
  order: 'descending',
};

const useStyles = makeStyles({
  title: {
    marginBottom: '10px',
  },
  label: {
    marginBottom: '5px',
  },
});

const _getSortOpts = (config: WidgetConfig, defaultOpt: SortOption) => {
  const {dimensions = [], measures = []} = config;
  const opts: SortOption[] = [];
  if (!isRecordExist(config)) {
    opts.push(defaultOpt);
  }
  dimensions.forEach((cdd: Dimension) => {
    const {label, as} = cdd;
    opts.push({
      value: as,
      label,
      tip: 'A-Z',
    });
  });
  measures.forEach((cdd: Measure) => {
    const {label, as, expression} = cdd;
    opts.push({
      value: as,
      label,
      tip: expression || '',
    });
  });
  return opts;
};

type SortProps = {
  config: WidgetConfig;
  setConfig: Function;
};
const Sort: FC<SortProps> = props => {
  const {nls} = useContext(I18nContext);
  const classes = useStyles({});
  const {config, setConfig} = props;
  let {sort = DEFAULT_SORT, measures = []} = config;
  const recordMeasure = measures.find((m: Measure) => m.isRecords);
  const defaultOpt = {
    value: recordMeasure ? recordMeasure.as : 'countval',
    label: nls.label_records,
    tip: '',
  };
  const sortOpts = _getSortOpts(config, defaultOpt);
  const order = sort.order === 'descending' ? nls.label_desc : nls.label_asc;

  const changeSort = (value: string) => {
    if (sort.name === value) {
      return;
    }
    setConfig({
      type: CONFIG.ADD_SORT,
      payload: {
        order: sort.order,
        name: value,
      },
    });
  };

  const changeOrder = () => {
    const val: string = sort.order === 'ascending' ? 'descending' : 'ascending';
    setConfig({
      type: CONFIG.ADD_SORT,
      payload: {
        name: sort.name,
        order: val,
      },
    });
  };

  const currOpt = sortOpts.find((option: SortOption) => option.value === sort.name) || defaultOpt;

  useEffect(() => {
    if (!config.sort) {
      setConfig({type: CONFIG.ADD_SORT, payload: DEFAULT_SORT});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <div>
      <div className={classes.title}>{nls.label_sort_by}</div>
      <Selector currOpt={currOpt} options={sortOpts} onOptionChange={changeSort} />
      <Button onClick={changeOrder}>{order}</Button>
    </div>
  );
};

export default Sort;
