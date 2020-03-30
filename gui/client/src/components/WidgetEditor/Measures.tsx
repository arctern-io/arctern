import React, {FC, Fragment, useState, useEffect, useContext} from 'react';
import {queryContext} from '../../contexts/QueryContext';
import {I18nContext} from '../../contexts/I18nContext';
import MeasureSelector from './MeasureSelector';
import NoSelector from './NoSelector';
import {MeasuresProps} from '../../types';
import {RequiredType} from '../../utils/Consts';
import {Column, getValidColumns} from '../../utils/EditorHelper';
import {WIDGET, CONFIG} from '../../utils/Consts';
import {dimensionGetter, measureGetter} from '../../utils/WidgetHelpers';

const Measures: FC<MeasuresProps> = props => {
  const {nls} = useContext(I18nContext);
  const reqContext = useContext(queryContext);
  const {config, setConfig, measuresSetting, options} = props;
  const {dimensions = [], measures = [], type} = config;
  const firstMeasureSetting = (measuresSetting && measuresSetting[0]) || {};
  const firstMeasureSettingType = firstMeasureSetting.type;
  const [isEnableAddMore, setEnableAddMore] = useState(true);
  useEffect(() => {
    switch (type) {
      case WIDGET.LINECHART:
        const isColorDimensinoExist = Boolean(dimensionGetter(config, 'color'));
        const _isEnableAddMore = !isColorDimensinoExist || measures.length === 0;
        setEnableAddMore(_isEnableAddMore);
        break;

      default:
        setEnableAddMore(true);
        break;
    }
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [type, dimensions]);

  // Required Measure Add
  const addMeasure = (measure: any, onAdd: any) => {
    measure.name = measure.name || `y${measures.length}`;
    onAdd
      ? onAdd({
          measure,
          config,
          setConfig,
          reqContext,
        })
      : setConfig({payload: measure, type: CONFIG.ADD_MEASURE});
  };

  const deleteMeasure = (as: string, onDelete: Function | undefined) => {
    const target = config.measures.find((m: any) => m.as === as);
    setConfig({payload: target, type: CONFIG.DEL_MEASURE});
    if (onDelete) {
      onDelete({config, measure: target, setConfig});
    }
  };

  let validColumns: Column[] = getValidColumns(options, firstMeasureSetting.columnTypes!);
  switch (firstMeasureSettingType) {
    case RequiredType.REQUIRED:
      return (
        <div className="measuresAll">
          {measuresSetting.map((requiredMeasure: any) => {
            const validColumns = getValidColumns(options, requiredMeasure.columnTypes);
            const measure = measureGetter(config, requiredMeasure.key);
            return (
              <Fragment key={requiredMeasure.key}>
                <MeasureSelector
                  setting={requiredMeasure}
                  placeholder={`+ ${nls.label_add_measure}`}
                  measure={measure}
                  options={validColumns}
                  addMeasure={addMeasure}
                  deleteMeasure={deleteMeasure}
                />
              </Fragment>
            );
          })}
        </div>
      );
    case RequiredType.ANY:
    case RequiredType.REQUIRED_ONE_DIMENSION_OR_MEASURE_AT_LEAST:
    case RequiredType.REQUIRED_ONE_AT_LEAST:
      return (
        <div className="measuresAll">
          {measures.map((measure: any, index: number) => {
            if (
              firstMeasureSettingType === RequiredType.REQUIRED_ONE_DIMENSION_OR_MEASURE_AT_LEAST
            ) {
              dimensions.length === 0 && delete measure.expression;
            }
            return (
              <Fragment key={measure.as}>
                <MeasureSelector
                  options={validColumns}
                  measure={measure}
                  setting={{
                    ...firstMeasureSetting,
                    labelPlus: (index + 1).toString(),
                  }}
                  addMeasure={addMeasure}
                  deleteMeasure={deleteMeasure}
                />
              </Fragment>
            );
          })}
          <MeasureSelector
            placeholder={`+ ${nls.label_add_measure}`}
            setting={firstMeasureSetting}
            options={validColumns}
            addMeasure={addMeasure}
            mLength={measures.length}
            isEnableAddMore={isEnableAddMore}
          />
        </div>
      );
    case RequiredType.NONE_REQUIRED:
    default:
      return <NoSelector />;
  }
};

export default Measures;
