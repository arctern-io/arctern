import React, {useContext, Fragment} from 'react';
import {SimpleSelector as Selector} from '../common/selectors';
import {makeStyles} from '@material-ui/core/styles';
import {DATE_FORMAT, NUM_FORMAT, DATE_UNIT_LEVEL} from '../../utils/Formatters';
import {cloneObj} from '../../utils/Helpers';
import {isTextCol, isDateCol} from '../../utils/ColTypes';
import {I18nContext} from '../../contexts/I18nContext';
import {CONFIG} from '../../utils/Consts';

const useStyles = makeStyles({
  title: {
    marginBottom: '10px',
  },
  label: {
    marginBottom: '5px',
  },
});

const DimensionFormat = (props: any) => {
  const {nls} = useContext(I18nContext);
  const classes = useStyles({});
  const {config, setConfig} = props;
  const cloneConfig = cloneObj(config);
  const {dimensions = []} = cloneConfig;
  const validDimensions = dimensions.filter(
    (dimension: any) => !isTextCol(dimension.type) && !dimension.extract
  );
  if (validDimensions.length === 0) {
    return <></>;
  }

  const renderDimensions = validDimensions.map((dimension: any) => {
    const {label, as, type, format, extract, timeBin} = dimension;
    const isDateType = isDateCol(type) && !extract;
    //Attention: dimension timeBin unit must be smaller or equal to dimension format, ahhhhhhh
    let validFormatOpts: any[];
    if (isDateType) {
      validFormatOpts = DATE_FORMAT.filter(
        (opt: any) => DATE_UNIT_LEVEL[timeBin] <= DATE_UNIT_LEVEL[opt.unit]
      );
    } else {
      validFormatOpts = NUM_FORMAT;
    }
    const currentFormatOpt: any =
      validFormatOpts.filter((opt: any) => opt.value === format)[0] || validFormatOpts[0];
    return {
      label,
      validFormatOpts,
      currentFormatOpt,
      as,
    };
  });

  const onFormatChange = (val: string, as: any) => {
    const target = config.dimensions.find((d: any) => d.as === as);
    target.format = val;
    setConfig({payload: {dimension: target}, type: CONFIG.ADD_DIMENSION});
  };

  return (
    <div>
      <div className={classes.title}>{nls.label_dimension_date_format}</div>
      {renderDimensions.map((dimension: any, index: number) => {
        const {label, validFormatOpts, currentFormatOpt, as} = dimension;
        return (
          <Fragment key={index}>
            <div className={classes.label}>
              <strong>
                {nls[`label_widgetEditor_binOpt_${currentFormatOpt.label} `]}
                {label}
              </strong>
            </div>
            <Selector
              currOpt={currentFormatOpt}
              options={validFormatOpts}
              onOptionChange={(format: any) => {
                onFormatChange(format, as);
              }}
            />
          </Fragment>
        );
      })}
    </div>
  );
};

export default DimensionFormat;
