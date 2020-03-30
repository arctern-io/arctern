import React, {useContext} from 'react';
import {useTheme, makeStyles} from '@material-ui/core/styles';
import Divider from '@material-ui/core/Divider';
import EditableLabel from '../common/EditableLabel';
import {I18nContext} from '../../contexts/I18nContext';
import {CONFIG} from '../../utils/Consts';
import WidgetPlaceholder from './WidgetPlaceholder';
import {getWidgetTitle} from '../../utils/WidgetHelpers';
import {genWidgetEditorStyle} from './index.style';
import './index.scss';
const useStyles = makeStyles(theme => genWidgetEditorStyle(theme) as any) as Function;
const WidgetView = (props: any) => {
  const {nls} = useContext(I18nContext);
  const theme = useTheme();
  const classes = useStyles(theme);
  const {config, setConfig, isReady, width, height, Widget, dimensionsReady, measuresReady} = props;
  const onTitleChange = ({title}: any) => {
    setConfig && setConfig({type: CONFIG.UPDATE_TITLE, payload: title});
  };

  const onClearFilter = () => {
    setConfig && setConfig({type: CONFIG.CLEAR_FILTER});
  };

  return (
    <>
      <div className={classes.widgetTitle}>
        <EditableLabel
          onChange={onTitleChange}
          label={getWidgetTitle(config, nls)}
          labelClass={classes.widgetTitle}
        />
        <Divider />
        <div
          className={classes.customDeleteIcon}
          onClick={onClearFilter}
          dangerouslySetInnerHTML={{
            __html: `<svg class="icon" viewBox="0 0 48 48"><polygon style="fill: currentColor" points="46,29.9 44.1,28 40,32.2 35.9,28 34,29.9 38.2,34 34,38.1 35.9,40 40,35.8 44.1,40 46,38.1 41.8,34  "></polygon><g id="icon-filter"><path fill=currentColor d="M40,6.5H8L6.8,8.9l11.7,15.6V44v2.4l2.2-1.1l0,0l8-4l0,0l0.8-0.4V40V24.5L41.2,8.9L40,6.5z M25,23v16.8l-3.5,1.8V24v-0.5 l-0.3-0.4l0,0l-5.4-7.3L11,9.5h24.1L25,23z"></path></g></svg>`,
          }}
        />
      </div>
      <div className={`${classes.widgetContent}`}>
        {isReady && <Widget {...props} wrapperWidth={width} wrapperHeight={height} />}
        {!isReady && (
          <WidgetPlaceholder
            {...props}
            dimensionsReady={dimensionsReady}
            measuresReady={measuresReady}
          />
        )}
      </div>
    </>
  );
};

export default WidgetView;
