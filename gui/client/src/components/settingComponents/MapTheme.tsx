import React, {useContext, useEffect} from 'react';
import {makeStyles} from '@material-ui/core/styles';
import {SimpleSelector as Selector} from '../common/selectors';
import {I18nContext} from '../../contexts/I18nContext';
import {MapThemes, DefaultMapTheme} from '../../widgets/Utils/Map';
import {CONFIG} from '../../utils/Consts';

const useStyles = makeStyles({
  title: {
    marginBottom: '10px',
  },
  label: {
    marginBottom: '5px',
  },
});

const MapTheme = (props: any) => {
  const {nls} = useContext(I18nContext);
  const {config, setConfig} = props;
  const classes = useStyles({});
  const {mapTheme} = config;
  const currOpt = MapThemes.find((item: any) => item.value === mapTheme) || MapThemes[0];
  const type = CONFIG.ADD_MAPTHEME;
  const onMapThemeChange = (value: string) => {
    setConfig({payload: value, type});
  };

  useEffect(() => {
    if (!config.mapTheme) {
      setConfig({payload: DefaultMapTheme, type});
    }
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div>
      <div className={classes.title}>{nls.label_widgetEditorDisplay_mapTheme}</div>
      <Selector currOpt={currOpt} options={MapThemes} onOptionChange={onMapThemeChange} />
    </div>
  );
};

export default MapTheme;
