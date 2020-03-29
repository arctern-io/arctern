import React, {useState, ChangeEvent, FocusEvent, KeyboardEvent} from 'react';
import {useTheme, makeStyles} from '@material-ui/core/styles';
import Input from '@material-ui/core/Input';
import {MODE} from '../../utils/Consts';
import {customOptsStyle} from '../../utils/Theme';

const useStyles = makeStyles(theme => ({
  ...customOptsStyle,
  chip: {
    margin: '0 5px',
  },
  title: {
    flexGrow: 1,
    textAlign: 'center',
    margin: 0,
    padding: 0,
  },
  titleInput: {
    fontSize: '2em',
    textAlign: 'center',
    fontWeight: 'bold',
    cursor: 'pointer',
    marginRight: theme.spacing(2),
  },
}));

interface IEditableLabel {
  label: string;
  labelClass?: string;
  onChange: (params: {title: string}) => void;
}

const EditableLabel = (props: IEditableLabel) => {
  const theme = useTheme();
  const classes = useStyles(theme) as any;
  const {label = '', onChange, labelClass = classes.titleInput} = props;
  const [mode, setMode] = useState<MODE>(MODE.NORMAL);
  const [_label, setLabel] = useState<string>(label);

  const onClick = () => {
    setMode(MODE.EDIT);
  };

  const onInputChange = (e: ChangeEvent<HTMLInputElement>) => setLabel(e.target.value);

  const onChangeLabel = (e: FocusEvent<HTMLInputElement>) => {
    if (e.target.value !== '') {
      onChange({title: e.currentTarget.value});
    }
    setMode(MODE.NORMAL);
  };

  const onKeyUp = (e: KeyboardEvent<HTMLInputElement>) => {
    // enter
    if (e.keyCode === 13) {
      onChange({title: e.currentTarget.value});
      setMode(MODE.NORMAL);
    }
    // escape
    if (e.keyCode === 27) {
      setMode(MODE.NORMAL);
    }
  };

  return (
    <>
      {mode === MODE.NORMAL && (
        <div className={labelClass} onClick={onClick}>
          {label}
        </div>
      )}
      {mode === MODE.EDIT && (
        <Input
          classes={{input: labelClass}}
          value={_label}
          autoFocus
          disableUnderline
          onChange={onInputChange}
          onKeyUp={onKeyUp}
          onBlur={onChangeLabel}
        />
      )}
    </>
  );
};

export default EditableLabel;
