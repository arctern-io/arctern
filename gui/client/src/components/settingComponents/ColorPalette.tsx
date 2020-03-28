import React, {useState, useEffect, useRef, useContext} from 'react';
import clsx from 'clsx';
import {I18nContext} from '../../contexts/I18nContext';
import {makeStyles, useTheme} from '@material-ui/core/styles';
import Card from '@material-ui/core/Card';
import {solidOpts, ordinalOpts, gradientOpts, isGradientType} from '../../utils/Colors';
import {genEffectClickOutside} from '../../utils/EditorHelper';
import {measureGetter} from '../../utils/WidgetHelpers';
import Ruler from './Ruler';
import {
  colorItemHeight,
  colorItemSelectedPadding,
  titleMarginBottom,
  subTitleMarginBottom,
} from '../../utils/Theme';
import {CONFIG} from '../../utils/Consts';

const useStyles = makeStyles(theme => ({
  title: {
    marginBottom: titleMarginBottom,
  },
  subTitle: {
    marginBottom: subTitleMarginBottom,
  },
  label: {
    marginBottom: '5px',
  },
  allOpts: {
    padding: '10px',
    borderColor: theme.palette.grey[700],
    border: 'solid',
    borderWidth: '.5px',
  },
  currRoot: {
    display: 'flex',
    justifyContent: 'start',
    alignItems: 'center',
  },
  optsRoot: {
    marginBottom: '5px',
  },
  currSolid: {
    width: colorItemHeight,
    height: colorItemHeight,
    marginBottom: '5px',
    cursor: 'pointer',
  },
  currScale: {
    flexGrow: 1,
    height: colorItemHeight,
  },
  selected: {
    border: 'solid',
    borderWidth: '3px',
    borderColor: 'white',
  },
  optContainer: {
    display: 'flex',
    marginBottom: '5px',
    cursor: 'pointer',
  },
  colorItem: {
    marginRight: '2px',
    marginBottom: '2px',
    height: colorItemHeight,
    width: colorItemHeight,
    padding: colorItemSelectedPadding,
    cursor: 'pointer',
  },
  growColorItem: {
    flexGrow: 1,
    height: colorItemHeight,
  },
}));

const ColorPalette = (props: any) => {
  const {nls} = useContext(I18nContext);
  const theme = useTheme();
  const classes = useStyles(theme);
  const {
    config,
    setConfig,
    customColorOpts = gradientOpts,
    colorTypes = [], //solid, ordinal, gradient
    isUseRuler = isGradientType(config.colorKey),
  } = props;
  const {colorKey = ''} = config;
  const colorMeasure = measureGetter(config, 'color');
  const root = useRef<HTMLDivElement>(null);
  const [isShowOpts, setShowOpts] = useState(false);

  const title = `${nls.label_color_palette}${colorMeasure ? `: ${colorMeasure.label}` : ''}`;
  const _genOpts = (colorTypes: any[]) => {
    const [useSolid, useOrdinal, useGradient] = [
      !!colorTypes.find((opt: string) => opt === 'solid'),
      !!colorTypes.find((opt: string) => opt === 'ordinal'),
      !!colorTypes.find((opt: string) => opt === 'gradient'),
    ];
    return (
      <>
        {useSolid && _genSolidOpts(solidOpts, onColorKeyChange)}
        {useOrdinal && _genOrdinalOpts(ordinalOpts, onColorKeyChange)}
        {useGradient && _genGradientOpts(customColorOpts, onColorKeyChange)}
      </>
    );
  };
  const _getColorValue = (colorKey: string, opts: any[]) =>
    opts.find((opt: any) => opt.key === colorKey);
  const _genCurrOptEle = (colorKey: string) => {
    const grandientOpt = _getColorValue(colorKey, gradientOpts);
    const ordinalOpt = _getColorValue(colorKey, ordinalOpts);
    const solidOpt = _getColorValue(colorKey, solidOpts);
    return (
      <div className={classes.currRoot} onClick={() => setShowOpts(true)}>
        {grandientOpt && (
          <div
            className={classes.growColorItem}
            key={colorKey}
            style={{
              background: `linear-gradient(to right, ${grandientOpt.value.join(', ')})`,
            }}
          />
        )}
        {ordinalOpt &&
          ordinalOpt.value.map((color: any) => {
            return (
              <div className={classes.currScale} style={{backgroundColor: color}} key={color} />
            );
          })}
        {solidOpt && (
          <div
            className={classes.currSolid}
            style={{backgroundColor: solidOpt.value}}
            data-value={solidOpt.value}
          />
        )}
      </div>
    );
  };

  const _genSolidOpts = (opts: any, onClick?: any) => {
    return (
      <div className={classes.optsRoot}>
        <div className={classes.optContainer}>
          {opts.map((opt: any) => {
            const isSelected = opt.key === colorKey;
            return (
              <div
                className={clsx(classes.colorItem, isSelected ? classes.selected : '')}
                onClick={onClick}
                key={opt.key}
                style={{
                  backgroundColor: opt.value,
                }}
                data-key={opt.value}
              />
            );
          })}
        </div>
      </div>
    );
  };

  const _genOrdinalOpts = (opts: any, onClick: any) => {
    return (
      <div className={classes.optsRoot}>
        {opts.map((opt: any) => {
          const isSelected = opt.key === colorKey;
          return (
            <div
              className={`${classes.optContainer} ${isSelected ? classes.selected : ''}`}
              key={opt.key}
              style={{display: 'flex', marginBottom: '5px'}}
              onClick={onClick}
              data-key={opt.key}
            >
              {opt.value.map((color: any) => {
                return (
                  <div
                    className={classes.growColorItem}
                    key={color}
                    style={{
                      backgroundColor: color,
                    }}
                  />
                );
              })}
            </div>
          );
        })}
      </div>
    );
  };

  const _genGradientOpts = (opts: any, onClick: any) => {
    return (
      <div className={classes.optsRoot}>
        {opts.map((opt: any) => {
          const isSelected = opt.key === colorKey;
          return (
            <div
              className={`${classes.optContainer} ${isSelected ? classes.selected : ''}`}
              key={opt.key}
              style={{display: 'flex', marginBottom: '5px'}}
              onClick={onClick}
              data-key={opt.key}
            >
              <div
                className={classes.growColorItem}
                key={opt.value[0]}
                style={{
                  background: `linear-gradient(to right, ${opt.value})`,
                }}
              />
            </div>
          );
        })}
      </div>
    );
  };

  const onColorKeyChange = (e: any) => {
    setConfig({payload: e.currentTarget.dataset.key, type: CONFIG.ADD_COLORKEY});
  };

  const _getColorOpts = (colorType: string) => {
    switch (colorType) {
      case 'ordinal':
        return ordinalOpts;
      case 'gradient':
        return gradientOpts;
      case 'solid':
      default:
        return solidOpts;
    }
  };
  const isValidColorKey = (colorKey: string | undefined, colorTypes: string[]) => {
    if (!colorKey) {
      return false;
    }
    let validOpts: any[] = [];
    colorTypes.forEach((c: string) => (validOpts = validOpts.concat(_getColorOpts(c))));
    return !!validOpts.find((opt: any) => opt.key === colorKey);
  };

  const _genDefaultColorKey = (colorTypes: string[]) => _getColorOpts(colorTypes[0])[0].key;

  // hide when click outside
  useEffect(() => {
    const hideDimension = genEffectClickOutside(root.current, setShowOpts, false);
    document.body.addEventListener('click', hideDimension);
    return () => {
      document.body.removeEventListener('click', hideDimension);
    };
  }, []);

  useEffect(() => {
    if (!isValidColorKey(colorKey, colorTypes)) {
      setConfig({payload: _genDefaultColorKey(colorTypes), type: CONFIG.ADD_COLORKEY});
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [JSON.stringify(colorTypes)]);

  return (
    <div ref={root}>
      <div className={classes.title}>{title}</div>
      {_genCurrOptEle(colorKey)}
      {isShowOpts && <Card classes={{root: classes.allOpts}}>{_genOpts(colorTypes)}</Card>}
      {isUseRuler && <Ruler {...props} />}
    </div>
  );
};

export default ColorPalette;
