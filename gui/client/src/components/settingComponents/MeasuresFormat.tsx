import React, {FC, useContext, Fragment} from 'react';
import {SimpleSelector as Selector} from '../common/selectors';
import {makeStyles} from '@material-ui/core/styles';
import {NUM_FORMAT} from '../../utils/Formatters';
import {I18nContext} from '../../contexts/I18nContext';
import {CONFIG} from '../../utils/Consts';
import {Measure, WidgetConfig} from '../../types';
import {getDefaultTitle} from '../../utils/EditorHelper';

const useStyles = makeStyles(theme => ({
  title: {
    marginBottom: theme.spacing(2),
  },
  label: {
    marginBottom: theme.spacing(1),
  },
}));

type MeasuresFormatProps = {
  config: WidgetConfig;
  setConfig: Function;
  measure: Measure;
};
const MeasuresFormat: FC<MeasuresFormatProps> = props => {
  const {nls} = useContext(I18nContext);
  const classes = useStyles();
  const {config, setConfig} = props;
  const {measures} = config;
  if (measures.length === 0) {
    return <></>;
  }
  const getTitle = (measure: Measure) => {
    const {expression, label} = getDefaultTitle(measure);
    return `${nls[`label_widgetEditor_expression_${expression}`] || ''} ${label}`;
  };

  const onFormatChange = (format: string, val: string) => {
    const measure = config.measures.find((measure: Measure) => measure.as === val)!;
    measure.format = format;
    setConfig({type: CONFIG.ADD_MEASURE, payload: measure});
  };
  return (
    <div>
      <div className={classes.title}>{nls.label_measure_number_format}</div>
      {measures.map((_measure: Measure) => {
        const currOpt = NUM_FORMAT.find(
          (item: {[key: string]: string}) => item.value === _measure.format
        ) || {value: 'auto', label: 'auto'};
        return (
          <Fragment key={_measure.as}>
            <div className={classes.label}>
              <strong>{getTitle(_measure)}</strong>
            </div>
            <Selector
              currOpt={currOpt}
              options={NUM_FORMAT}
              onOptionChange={(format: string) => {
                onFormatChange(format, _measure.as);
              }}
            />
          </Fragment>
        );
      })}
    </div>
  );
};

export default MeasuresFormat;
