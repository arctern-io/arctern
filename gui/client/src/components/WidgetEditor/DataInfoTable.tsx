import React, {useContext, useRef} from 'react';
import {I18nContext} from '../../contexts/I18nContext';
import Table from '../common/Table';
import {formatSource} from '../../utils/Helpers';
import {genWidgetEditorStyle} from './index.style';
import {useTheme, makeStyles} from '@material-ui/core/styles';
import './index.scss';
const useStyles = makeStyles(theme => genWidgetEditorStyle(theme) as any) as Function;

const DataInfoTable = (props: any) => {
  const {nls} = useContext(I18nContext);
  const theme = useTheme();
  const classes = useStyles(theme);
  const {dashboard, title} = props;
  const {sourceOptions, totals} = dashboard;
  const baseInfoNode = useRef<HTMLDivElement>(null);
  const def = [
    {field: 'col_name', name: nls.label_colName},
    {field: 'data_type', name: nls.label_column_type},
  ];

  return (
    <div ref={baseInfoNode} className={`${classes.colNamePreview}`}>
      <div className={classes.baseInfo}>
        <div className={classes.baseInfoWrapper}>
          <div className={classes.baseInfoTitle}>{nls.label_dataList_abstract_tableName}</div>
          <h1 className={classes.baseInfoDetail}>{formatSource(title)}</h1>
        </div>
        <div className={classes.baseInfoWrapper}>
          <div className={classes.baseInfoTitle}>{nls.label_dataList_abstract_numRows}</div>
          <h1 className={classes.baseInfoDetail}>{totals[title]}</h1>
        </div>
        <div className={classes.baseInfoWrapper}>
          <div className={classes.baseInfoTitle}>{nls.label_dataList_abstract_columns}</div>
          <h1 className={classes.baseInfoDetail}>{sourceOptions[title].length || 0}</h1>
        </div>
      </div>
      <div className={classes.colNames}>
        <Table
          data={sourceOptions[title] || []}
          def={def}
          length={(sourceOptions[title] || []).length}
          isUsePagination={false}
        />
      </div>
    </div>
  );
};

export default DataInfoTable;
