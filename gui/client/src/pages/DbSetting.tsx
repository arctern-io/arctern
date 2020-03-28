import React, {FC, useState, useContext, useEffect} from 'react';
import {Redirect, RouteComponentProps} from 'react-router-dom';
import {makeStyles} from '@material-ui/core/styles';
import CssBaseline from '@material-ui/core/CssBaseline';
import Divider from '@material-ui/core/Divider';
import Container from '@material-ui/core/Container';
import Button from '@material-ui/core/Button';
import MenuItem from '@material-ui/core/MenuItem';
import Select from '@material-ui/core/Select';
import Spinner from '../components/common/Spinner';
import {authContext} from '../contexts/AuthContext';
import {I18nContext} from '../contexts/I18nContext';
import {queryContext} from '../contexts/QueryContext';
import {rootContext} from '../contexts/RootContext';
import {DB_TYPE} from '../types';

const useStyles = makeStyles(() => ({
  dataBaseConfiger: {
    flexGrow: 1,
    display: 'flex',
    flexDirection: 'column',
  },
  customContainer: {
    flexGrow: 1,
    display: 'flex',
  },
  title: {
    paddingLeft: '20px',
    width: '100%',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    flexDirection: 'column',
  },
  customDivider: {
    marginBottom: '20px',
  },
  link: {
    color: 'white',
    textDecoration: 'none',
  },
  paper: {
    flexGrow: 1,
    display: 'flex',
  },
  form: {
    paddingTop: '20px',
    flexGrow: 1,
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'start',
    alignItems: 'left',
  },
  buttonList: {
    margin: '16px 0 20px 0',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  submit: {},
}));

const DbSetting: FC<RouteComponentProps> = props => {
  const classes = useStyles();
  const {auth} = useContext(authContext);
  const {getDBs, DB = false, setDB} = useContext(queryContext);
  const {setDialog} = useContext(rootContext);
  const {nls} = useContext(I18nContext);
  const [loading, setLoading] = useState(true);
  const [localDb, setLocalDb] = useState<DB_TYPE | false>(DB);
  const [dbs, setDBs] = useState<DB_TYPE[]>([]);

  useEffect(() => {
    if (auth.userId === 'guest' || nls === null) {
      return;
    }
    // get connection information
    getDBs().then((res: any) => {
      if (res.data) {
        setDBs(res.data.data);
        if (localDb === false) {
          setLocalDb(res.data.data[0]);
        }
        setLoading(false);
      } else {
        setDB(false);
      }
    });
    //eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auth.userId, nls]);

  const onTypeChange = (e: any) => {
    const target = dbs.find((d: DB_TYPE) => d.type === e.target.value)!;
    setLocalDb(target);
  };

  const onDBChange = (e: any) => {
    const target = dbs.find((d: DB_TYPE) => d.id === e.target.value)!;
    setLocalDb(target);
  };

  const onSubmit = () => {
    setDB(localDb);
    setDialog({
      open: true,
      title: nls.label_db_success_title,
      content: nls.label_db_success_content,
      onConfirm: () => {
        props.history.push('/');
      },
    });
  };

  const useDefaultConfig = () => {
    setLocalDb(dbs[0]);
  };

  if (auth.userId === 'guest') {
    return <Redirect to="/login" />;
  }

  if (loading) {
    return <Spinner />;
  }
  const curr = localDb || dbs[0];
  const cdds: DB_TYPE[] = dbs.filter((d: DB_TYPE) => d.type === curr.type);
  return (
    <div className={classes.dataBaseConfiger}>
      <div className={classes.title}>
        <h1>{nls.label_db_title}</h1>
      </div>
      <Divider classes={{root: classes.customDivider}} />
      <Container classes={{root: classes.customContainer}} component="main" maxWidth="xs">
        <CssBaseline />
        <div className={classes.paper}>
          <div className={classes.form}>
            <Select label={'DbType'} value={curr.type} onChange={onTypeChange}>
              {dbs.map((db: DB) => (
                <MenuItem value={db.type} key={db.type}>
                  {db.type}
                </MenuItem>
              ))}
            </Select>
            <Select value={curr.id} onChange={onDBChange}>
              {cdds.map((cdd: DB) => (
                <MenuItem value={cdd.id} key={cdd.id}>
                  {cdd.name}
                </MenuItem>
              ))}
            </Select>
            <div className={classes.buttonList}>
              <Button
                size="medium"
                variant="outlined"
                className={classes.submit}
                onClick={onSubmit}
              >
                {nls.label_db_save}
              </Button>
              <Button size="medium" variant="contained" onClick={useDefaultConfig}>
                {nls.label_db_useDefault}
              </Button>
            </div>
          </div>
        </div>
      </Container>
    </div>
  );
};

export default DbSetting;
