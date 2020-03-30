import React, {useContext} from 'react';
import {I18nContext} from '../../contexts/I18nContext';
import clsx from 'clsx';
import {genWidgetEditorStyle} from './index.style';
import {useTheme, makeStyles} from '@material-ui/core/styles';
import './index.scss';
const useStyles = makeStyles(theme => genWidgetEditorStyle(theme) as any) as Function;

const WidgetPlaceholder = (props: any) => {
  const {language, nls} = useContext(I18nContext);
  const theme = useTheme();
  const classes = useStyles(theme);
  const {dimensionsReady, measuresReady, config} = props;

  return (
    <div className={clsx(classes.requirement, `${config.type}-${language}`)}>
      <h1 className={classes.label}>
        {nls[`label_Header_${config.type}`] || config.type} {nls.label_oror_requirements}
      </h1>
      {dimensionsReady.lacks.map((lack: any, index: number) => (
        <h2 className={classes.requirement} key={index}>
          {nls[`label_widgetEditor_requireLabel_${lack.name}`]} {nls.label_dimensions}
        </h2>
      ))}
      {JSON.stringify(dimensionsReady.lacks) === JSON.stringify(measuresReady.lacks) && (
        <h4 style={{textAlign: 'center'}}>{nls.label_or}</h4>
      )}
      {measuresReady.lacks.map((lack: any, index: number) => (
        <h2 className={classes.requirement} key={index}>
          {nls[`label_widgetEditor_requireLabel_${lack.name}`]} {nls.label_measures}
        </h2>
      ))}
    </div>
  );
};

export default WidgetPlaceholder;
