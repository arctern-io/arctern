import React, {FC, useContext, useEffect} from 'react';
import clsx from 'clsx';
import {withRouter} from 'react-router-dom';
import {makeStyles, useTheme, Theme} from '@material-ui/core/styles';
import {
  Dashboard as DashboardIcon,
  Settings as SettingsIcon,
  AccountBox as AccountBoxIcon,
} from '@material-ui/icons';
import {Drawer, IconButton} from '@material-ui/core';
import Logo from '../../logo.svg';
import {rootContext} from '../../contexts/RootContext';
import {authContext} from '../../contexts/AuthContext';
import {genBasicStyle} from '../../utils/Theme';
const useStyles = makeStyles((theme: Theme) => ({
  ...genBasicStyle(theme.palette.primary.main),
  paper: {
    backgroundColor: '#000',
  },
  muiRoot: {
    width: theme.spacing(8),
  },
  logo: {
    background: `url(${Logo}) no-repeat center`,
    width: theme.spacing(8),
    height: '100px',
    marginBottom: theme.spacing(2),
    cursor: 'pointer',
  },
  container: {
    height: '100%',
    display: 'flex',
    flexDirection: 'column',
    justifyContent: 'space-between',
  },
  wrapper: {
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center',
    width: '100%',
    color: '#fff',
    fontSize: '3px',
    paddingBottom: theme.spacing(2),
  },
  selected: {
    color: theme.palette.primary.main,
  },
  icon: {
    fontSize: '2rem',
    marginBottom: theme.spacing(1),
  },
  settingIcon: {
    fontSize: '1.5rem',
  },
}));

const Nav: FC<any> = (props: any) => {
  const {auth} = useContext(authContext);
  const {theme} = useContext(rootContext);
  const _theme = useTheme();
  const {onAvatarClick} = props;
  const isDemo = auth.userId === 'demo';
  const classes = useStyles(_theme);
  let path = props.history.location.pathname;

  useEffect(() => {
    document.body.setAttribute('style', `background-color: ${_theme.palette.background.default};`);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);
  return (
    <Drawer
      className={classes.muiRoot}
      classes={{paper: classes.paper}}
      variant="permanent"
      anchor="left"
    >
      <div className={classes.logo} onClick={() => props.history.push('/')}></div>
      <div className={classes.container}>
        <div className={clsx(classes.wrapper, classes.hover, {[classes.selected]: path === '/'})}>
          <DashboardIcon onClick={() => props.history.push('/')} classes={{root: classes.icon}} />
          <span>{`Dashboards`}</span>
        </div>
        <div>
          <div className={clsx(classes.wrapper, classes.hover)}>
            <IconButton disabled={isDemo} onClick={() => props.history.push('/config')}>
              <SettingsIcon classes={{root: classes.settingIcon}} />
            </IconButton>
          </div>
          <div className={clsx(classes.wrapper, classes.hover)}>
            <AccountBoxIcon onClick={onAvatarClick} classes={{root: classes.settingIcon}} />
          </div>
        </div>
      </div>
    </Drawer>
  );

  // <NativeSelect
  //   classes={{root: classes.naveSelect}}
  //   value={theme}
  //   onChange={(e: any) => {
  //     saveTheme(e.target.value);
  //   }}
  // >
  //   {themes.map((theme: string) => {
  //     return (
  //       <option value={theme} key={theme}>
  //         {theme}
  //       </option>
  //     );
  //   })}
  // </NativeSelect>;
};

export default withRouter(Nav);
