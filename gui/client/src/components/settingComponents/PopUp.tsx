import React, {Fragment, useContext, useEffect} from 'react';
import Card from '@material-ui/core/Card';
import {SimpleSelector as Selector} from '../common/selectors';
import {makeStyles, useTheme} from '@material-ui/core/styles';
import {I18nContext} from '../../contexts/I18nContext';
import {CONFIG} from '../../utils/Consts';

const useStyles = makeStyles(theme => ({
  allOpts: {
    padding: '10px',
    borderColor: theme.palette.grey[700],
    border: 'solid',
    borderWidth: '.5px',
  },
  options: {
    maxHeight: '150px',
    overflowY: 'scroll',
    margin: 0,
    padding: 0,
    listStyle: 'none',
  },
  option: {
    width: '100%',
    display: 'flex',
    position: 'relative',
  },
  title: {
    marginBottom: '10px !important',
  },
}));
const PopUp = (props: any) => {
  const {nls} = useContext(I18nContext);
  const theme = useTheme();
  const classes = useStyles(theme);
  const {config, setConfig, options} = props;
  const {popupItems = []} = config;

  const addPopupItem = (val: any) => {
    setConfig({type: CONFIG.ADD_POPUP_ITEM, payload: val});
  };

  const onDelete = (val: any) => {
    const target = popupItems.find((item: string) => item === val);
    setConfig({type: CONFIG.DEL_POPUP_ITEM, payload: target});
  };

  const onOptionChange = (val: string, option: string) => {
    if (val === option) {
      return;
    }
    setConfig({type: CONFIG.DEL_POPUP_ITEM, payload: option});
    setConfig({type: CONFIG.ADD_POPUP_ITEM, payload: val});
  };
  const filtedOpts = options
    .filter((val: string) => !popupItems.some((existOne: string) => existOne === val))
    .filter((val: string) => !config.measures.some((measure: any) => measure.value === val))
    .map((val: string) => {
      return {label: val, value: val};
    });

  useEffect(() => {
    if (!config.popupItems) {
      setConfig({type: CONFIG.ADD_POPUP_ITEM});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return (
    <>
      <p className={classes.title}>{nls.label_popup_box}</p>
      <Card classes={{root: classes.allOpts}}>
        {popupItems.map((option: string, index: number) => {
          return (
            <Fragment key={index}>
              <Selector
                currOpt={{value: option, label: option}}
                options={filtedOpts}
                onOptionChange={(val: string) => {
                  onOptionChange(val, option);
                }}
                onDelete={onDelete}
              />
            </Fragment>
          );
        })}
        <Selector
          currOpt={{value: ''}}
          options={filtedOpts}
          placeholder={`+ ${nls.label_add_column}`}
          onOptionChange={addPopupItem}
        />
      </Card>
    </>
  );
};

export default PopUp;
