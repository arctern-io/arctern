import React, {FC, useState, useContext, useEffect, useRef} from 'react';
import {RouteComponentProps} from 'react-router-dom';
import {timeFormat} from 'd3';
import {makeStyles} from '@material-ui/core/styles';
import Button from '@material-ui/core/Button';
import IconButton from '@material-ui/core/IconButton';
import AddBoxIcon from '@material-ui/icons/AddBox';
import InsertDriveFileIcon from '@material-ui/icons/InsertDriveFile';
import SaveAlt from '@material-ui/icons/SaveAlt';
import Delete from '@material-ui/icons/Delete';
import Spinner from '../components/common/Spinner';
import Table from '../components/common/Table';
import {authContext} from '../contexts/AuthContext';
import {I18nContext} from '../contexts/I18nContext';
import {queryContext} from '../contexts/QueryContext';
import {rootContext} from '../contexts/RootContext';
import {exportJson} from '../utils/Export';
import {cloneObj} from '../utils/Helpers';
import {DIALOG_MODE} from '../utils/Consts';
import {isDashboardReady} from '../utils/Dashboard';
import dashboards from '../mock';

const useStyles = makeStyles(theme => ({
  root: {
    flexGrow: 1,
    padding: theme.spacing(8),
  },
  marginRight: {
    marginRight: theme.spacing(1),
  },
  title: {
    paddingBottom: theme.spacing(4),
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  table: {
    minWidth: 650,
  },
  link: {
    color: 'white',
    textDecoration: 'none',
  },
}));

// table format
const format = (data: any[]) => {
  const formatter = timeFormat('%a %b %e %H:%M:%S %Y');
  return data.map((item: any) => {
    let cloneItem: any = {};
    Object.keys(item).forEach((key: string) => {
      if (key === 'createdAt' || key === 'modifyAt') {
        cloneItem[key] = formatter(new Date(item[key]));
      } else {
        cloneItem[key] = item[key];
      }
    });
    return cloneItem;
  });
};

let imported = false;

const Dashboards: FC<RouteComponentProps> = props => {
  const {history} = props;
  const {auth} = useContext(authContext);
  const {nls} = useContext(I18nContext);
  const {getDashboardList, saveDashboard, removeDashboard} = useContext(queryContext);
  const {widgetSettings, setDialog, setSnackbar} = useContext(rootContext);
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState<any>([]);
  const classes = useStyles();
  const fileBtn = useRef<HTMLInputElement>();

  useEffect(() => {
    let isSubscribed = true;
    getDashboardList(auth.userId).then((res: any) => {
      if (isSubscribed) {
        setLoading(false);
        setData(res);
      }
    });
    return () => {
      isSubscribed = false;
    };
  }, [auth, getDashboardList]);

  // delete a dashboard
  const deleteDashboard = (row: any) => {
    setDialog({
      open: true,
      mode: DIALOG_MODE.CONFIRM,
      title: nls.label_info,
      content: nls.tip_remove_dashboard,
      confirmLabel: nls.label_remove,
      onConfirm: () => {
        let cloneData = cloneObj(data);
        removeDashboard(row.id);
        let index = cloneData.findIndex((d: any) => d.id === row.id);
        if (index !== -1) {
          cloneData.splice(index, 1);
        }
        setData(cloneData);
      },
    });
  };

  // dashboard table list definition
  const def = [
    {
      field: 'id',
      name: nls.label_ID,
      format: null,
      onClick: (row: any) => {
        const value = `/bi/${row.id}`;
        history.push(value);
      },
    },
    {
      field: 'title',
      sortable: true,
      name: nls.label_name,
      onClick: (row: any) => {
        const value = `/bi/${row.id}`;
        history.push(value);
      },
    },
    {
      field: 'modifyAt',
      name: nls.label_modified,
    },
    {
      field: 'createdAt',
      name: nls.label_created,
    },
    {
      field: 'toolbar',
      name: '',
      widget: (row: any) => {
        const isEmbeded = row.demo;
        return (
          <>
            <IconButton
              title={nls.label_export}
              onClick={() => {
                exportJson(row);
              }}
            >
              <SaveAlt fontSize="small" />
            </IconButton>
            {!isEmbeded && (
              <IconButton
                title={nls.label_remove}
                onClick={() => {
                  deleteDashboard(row);
                }}
              >
                <Delete fontSize="small" />
              </IconButton>
            )}
          </>
        );
      },
    },
  ];

  // add new dashboard
  const onAddClicked = () => {
    let nextId: number = data.length + 1;
    let i = 1;
    // eslint-disable-next-line
    while (data.some((d: any) => d.id === nextId)) {
      nextId = data.length + i++;
    }
    const value = `/bi/${nextId}`;
    history.push(value);
  };

  // fire on click the import button
  const onImportClicked = () => {
    if (fileBtn.current) {
      fileBtn.current.click();
    }
  };

  const _import = (obj: any) => {
    let cloneData = cloneObj(data);
    obj.modifyAt = new Date().toUTCString();
    saveDashboard(JSON.stringify(obj), obj.id);
    let index = cloneData.findIndex((d: any) => d.id === obj.id);
    if (index !== -1) {
      cloneData[index] = obj;
    } else {
      cloneData.push(obj);
    }
    setData(cloneData);
    setSnackbar({open: true, message: nls.tip_import_dashboard_success});
  };

  // do the import
  const importDashboard = (obj: any) => {
    if (!isDashboardReady(obj, widgetSettings)) {
      setDialog({
        open: true,
        title: nls.label_info,
        content: nls.tip_dashboard_config_wrong,
      });
      return false;
    }

    // check conflict
    const conflict = data.filter((d: any) => d.id === obj.id)[0];

    if (conflict) {
      setDialog({
        open: true,
        mode: DIALOG_MODE.CONFIRM,
        title: nls.label_info,
        content: nls.tip_import_existed,
        onConfirm: () => {
          _import(obj);
        },
      });
    } else {
      _import(obj);
    }
  };

  // fire on file uploaded
  const onImportChange = (e: any) => {
    const fileReader = new FileReader();
    fileReader.onloadend = () => {
      let obj;
      try {
        obj = JSON.parse((fileReader.result as string) || '{}');
      } catch (error) {
        setDialog({
          open: true,
          title: nls.label_error,
          content: nls.tip_dashboard_config_wrong,
          onConfirm: () => {},
        });
      }
      if (obj && Object.keys(obj).length > 0) {
        importDashboard(obj);
        // need to clear the file input
        // so that we can upload file the next time
        if (fileBtn.current) {
          fileBtn.current.value = '';
        }
      } else {
        setDialog({
          open: true,
          title: nls.label_error,
          content: nls.tip_dashboard_config_wrong,
          onConfirm: () => {},
        });
      }
    };
    if (e.target.files.length > 0) {
      fileReader.readAsText(e.target.files[0]);
    }
  };

  useEffect(() => {
    if (!imported) {
      dashboards.map(importDashboard);
      imported = true;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  if (loading) {
    return <Spinner />;
  }

  return (
    <div className={classes.root}>
      <div className={classes.title}>
        <h1>{nls.label_saved_dashboard}</h1>
        <div>
          <Button
            variant="contained"
            color="primary"
            className={classes.marginRight}
            onClick={onAddClicked}
          >
            <AddBoxIcon className={classes.marginRight} fontSize="small" />
            {nls.label_dashboard_add}
          </Button>
          <input
            ref={(input: any) => (fileBtn.current = input)}
            type="file"
            onChange={onImportChange}
            hidden
          />
          <Button variant="contained" color="default" onClick={onImportClicked}>
            <InsertDriveFileIcon className={classes.marginRight} fontSize="small" />
            {nls.label_import_dashboard}
          </Button>
        </div>
      </div>
      <Table data={format(data)} def={def} length={data.length} />
    </div>
  );
};

export default Dashboards;
