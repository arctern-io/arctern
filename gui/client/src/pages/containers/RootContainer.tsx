import React, {FC, useContext} from 'react';
import {BrowserRouter, Route, Redirect, Switch} from 'react-router-dom';
import {createMuiTheme, makeStyles} from '@material-ui/core/styles';
import {ThemeProvider} from '@material-ui/styles';
import CssBaseline from '@material-ui/core/CssBaseline';
import InfoDialog from '../../components/common/Dialog';
import SnakeBar from '../../components/common/SnakeBar';
import {I18nContext} from './../../contexts/I18nContext';
import {authContext} from './../../contexts/AuthContext';
import Login from '../Login';
import Nav from '../../components/common/Nav';
import Spinner from '../../components/common/Spinner';
import {rootContext} from '../../contexts/RootContext';
import MainContainer from './MainContainer';
import DbSetting from '../DbSetting';

const RootContainer: FC = () => {
  const {nls} = useContext(I18nContext);
  const {auth, setUnauthStatus} = useContext(authContext);
  const {dialog, snakebar, theme, themeMap} = useContext(rootContext);
  const classes = makeStyles({
    root: {
      display: 'flex',
      height: '100%',
    },
  })();

  if (nls === null) {
    return (
      <div className="loading-container">
        <Spinner />
      </div>
    );
  }

  const isLogined = auth.userId !== 'guest';
  return (
    <div className={classes.root}>
      <ThemeProvider theme={createMuiTheme(themeMap[theme])}>
        <CssBaseline />
        <BrowserRouter>
          {isLogined && <Nav onAvatarClick={setUnauthStatus} />}
          <Switch>
            <Route path="/login" component={Login} />
            <Route path="/config" component={DbSetting} />
            <Route path="/" component={MainContainer} />
            <Route exact path="/bi">
              <Redirect to="/" />
            </Route>
          </Switch>
        </BrowserRouter>
        <SnakeBar {...snakebar} />
        <InfoDialog {...dialog} />
      </ThemeProvider>
    </div>
  );
};

export default RootContainer;
