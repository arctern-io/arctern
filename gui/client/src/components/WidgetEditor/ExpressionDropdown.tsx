import React, {FC, useContext} from 'react';
import Button from '@material-ui/core/Button';
import {I18nContext} from '../../contexts/I18nContext';
import {ExpressionDropdownProps} from '../../types';
import {makeStyles} from '@material-ui/core/styles';
import {DefaultExpressionOption, CustomExpressOption} from '../../utils/Consts';
import {genBasicStyle} from '../../utils/Theme';
import {cloneObj} from '../../utils/Helpers';

const useStyles = makeStyles(theme => ({
  ...genBasicStyle(theme.palette.primary.main),
  binRoot: {
    backgroundColor: theme.palette.background.default,
    width: '100%',
    maxWidth: '300px',
    padding: '0',
  },
  expression: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    flexWrap: 'wrap',
  },
  btn: {
    width: '125px',
    margin: '5px 0',
    fontSize: '13px',
  },
}));

const ExpressionDropdown: FC<ExpressionDropdownProps> = props => {
  const classes = useStyles({});
  const {nls} = useContext(I18nContext);
  const {measure, addMeasure, expressions} = props;
  const cloneMeasure = cloneObj(measure);
  const optList = expressions.map((exp: string) => {
    return {label: exp, value: exp};
  });
  const onClick = (val: string) => {
    cloneMeasure.expression = val;
    addMeasure(cloneMeasure);
  };
  return (
    <div className={classes.binRoot}>
      <div className={classes.expression}>
        {optList.map((option: DefaultExpressionOption | CustomExpressOption) => {
          const isSelected = option.value === measure.expression;
          return (
            <Button
              onClick={() => onClick(option.value)}
              className={classes.btn}
              variant={isSelected ? 'contained' : 'outlined'}
              color="secondary"
              key={option.value}
            >
              {nls[`label_widgetEditor_expression_${option.value}`] || option.label}
            </Button>
          );
        })}
      </div>
    </div>
  );
};

export default ExpressionDropdown;
