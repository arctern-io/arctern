import React, {FC, Fragment, useState, useEffect, useRef, useContext} from 'react';
import {I18nContext} from '../../contexts/I18nContext';
import {queryContext} from '../../contexts/QueryContext';
import {DimensionsProps, DimensionSetting, Dimension} from '../../types';
import DimensionSelector from './DimensionSelector';
import NoSelector from './NoSelector';
import {RequiredType, CONFIG} from '../../utils/Consts';
import {dimensionGetter} from '../../utils/WidgetHelpers';
import {Column, getValidColumns} from '../../utils/EditorHelper';
import {WIDGET} from '../../utils/Consts';

const Dimensions: FC<DimensionsProps> = props => {
  const {nls} = useContext(I18nContext);
  const reqContext = useContext(queryContext);
  const {config, setConfig, dimensionsSetting, options = []} = props;
  const {dimensions = [], measures = [], type = ''} = config;
  const firstDimensionSetting = dimensionsSetting[0]!;
  const firstRequiredType = firstDimensionSetting.type;
  const [length, setLength] = useState(0);
  const [isShowLast, setIsShowLast] = useState(false);
  const [isEnableColorDimension, setIsUseColorDimension] = useState(true);

  const addDimension = (dimension: any, onAdd: any) => {
    onAdd
      ? onAdd({
          dimension,
          config,
          setConfig,
          reqContext,
        })
      : setConfig({type: CONFIG.ADD_DIMENSION, payload: {dimension}});
  };

  const deleteDimension = (as: string, onDelete: Function | undefined) => {
    const target = dimensions.find((item: Dimension) => item.as === as);
    setConfig({type: CONFIG.DEL_DIMENSION, payload: {dimension: target}});
    if (onDelete) {
      onDelete({dimension: target, config, setConfig});
    }
  };

  const isFirstRun = useRef(true);
  useEffect(() => {
    if (isFirstRun.current) {
      isFirstRun.current = false;
      setLength(dimensions.length);
      return;
    }
    const _length = dimensions.length;
    switch (_length - length) {
      case 1:
        setIsShowLast(true);
        break;
      default:
        setIsShowLast(false);
        break;
    }
    setLength(_length);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dimensions.length]);

  useEffect(() => {
    switch (config.type) {
      case WIDGET.LINECHART:
        const isAddColorDimensionWork = measures.length <= 1;
        setIsUseColorDimension(isAddColorDimensionWork);
        break;
      default:
        setIsUseColorDimension(true);
        break;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [type, measures.length]);

  let validColumns: Column[] = getValidColumns(options, firstDimensionSetting.columnTypes!);
  switch (firstRequiredType) {
    case RequiredType.REQUIRED:
      return (
        <div className="dimensionsAll">
          {dimensionsSetting.map((dimensionSetting: DimensionSetting) => {
            const dimension = dimensionGetter(config, dimensionSetting.key!);
            const validColumns = getValidColumns(options, dimensionSetting.columnTypes!);
            const isColor = dimensionSetting.key === 'color';
            const enableAddColor = isEnableColorDimension || !isColor;
            return (
              <Fragment key={dimensionSetting.key}>
                <DimensionSelector
                  id={config.id}
                  source={config.source}
                  setting={dimensionSetting}
                  placeholder={`+ ${nls.label_add_dimension}`}
                  dimension={dimension}
                  options={validColumns}
                  deleteDimension={deleteDimension}
                  addDimension={addDimension}
                  enableAddColor={enableAddColor}
                  isShowLast={isShowLast}
                />
              </Fragment>
            );
          })}
        </div>
      );
    case RequiredType.REQUIRED_ONE_AT_LEAST:
    case RequiredType.REQUIRED_ONE_DIMENSION_OR_MEASURE_AT_LEAST:
    case RequiredType.ANY:
      return (
        <div className="dimensionsAll">
          {dimensions.map((dimension: Dimension) => {
            return (
              <Fragment key={dimension.as}>
                <DimensionSelector
                  id={config.id}
                  source={config.source}
                  dimension={dimension}
                  setting={firstDimensionSetting}
                  options={validColumns}
                  addDimension={addDimension}
                  deleteDimension={deleteDimension}
                  isShowLast={isShowLast}
                />
              </Fragment>
            );
          })}
          <DimensionSelector
            id={config.id}
            source={config.source}
            placeholder={`+ ${nls.label_add_dimension}`}
            options={validColumns}
            addDimension={addDimension}
            setting={firstDimensionSetting}
            dLength={dimensions.length}
          />
        </div>
      );
    case RequiredType.NONE_REQUIRED:
    default:
      return <NoSelector />;
  }
};

export default Dimensions;
