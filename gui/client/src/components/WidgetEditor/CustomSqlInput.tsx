import React, {useState, useContext} from 'react';
import {I18nContext} from '../../contexts/I18nContext';
import Button from '@material-ui/core/Button';
import TextField from '@material-ui/core/TextField';
import Card from '@material-ui/core/Card';
import {makeStyles, useTheme} from '@material-ui/core/styles';
const useStyles = makeStyles(theme => ({
  roots: {
    padding: '10px',
    borderColor: theme.palette.grey[700],
    border: 'solid',
    borderWidth: '.5px',
  },
  buttonList: {
    display: 'flex',
    justifyContent: 'space-between',
    marginBottom: '20px',
  },
  customInput: {
    marginBottom: '10px',
  },
}));

const CustomSqlInput = (props: any) => {
  const {nls} = useContext(I18nContext);
  const theme = useTheme();
  const classes = useStyles(theme);
  const {onCancel, onSave, placeholder, currVal} = props;
  const [label, setLabel] = useState(currVal.label || '');
  const [value, setValue] = useState(currVal.value || '');

  const onNameChange = (e: any) => {
    const val = e.target.value;
    setLabel(val);
  };

  const onFuncChange = (e: any) => {
    const val = e.target.value;
    setValue(val);
  };

  const onSaveClick = () => {
    onSave(label, value);
  };
  return (
    <Card classes={{root: classes.roots}}>
      <TextField
        required
        classes={{root: classes.customInput}}
        label={nls.label_widgetEditor_customOpt_label}
        placeholder={placeholder.name}
        fullWidth
        value={label}
        onChange={onNameChange}
      />
      <TextField
        required
        label={nls.label_widgetEditor_customOpt_expression}
        placeholder={placeholder.funcText}
        multiline
        rowsMax="4"
        value={value}
        onChange={onFuncChange}
        fullWidth
      />
      <div className={classes.buttonList}>
        <Button onClick={(e: any) => onCancel()}>{nls.label_widgetEditor_customOpt_back}</Button>
        <Button disabled={!value || !label} onClick={onSaveClick}>
          {nls.label_widgetEditor_customOpt_save}
        </Button>
      </div>
    </Card>
  );
};

export default CustomSqlInput;
