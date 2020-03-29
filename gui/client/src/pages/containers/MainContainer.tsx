import React, {FC, useContext} from 'react';
import {Route, Switch, Redirect} from 'react-router-dom';
import Dashboards from '../Dashboards';
import Bi from './Bi';
import Page404 from '../Page404';
import {authContext} from '../../contexts/AuthContext';
import {queryContext} from './../../contexts/QueryContext';

const MainContainer: FC<any> = () => {
  const {auth} = useContext(authContext);
  const {DB} = useContext(queryContext);
  if (auth.userId === 'guest') {
    return <Redirect to="/login" />;
  }
  if (DB === false) {
    return <Redirect to="/config" />;
  }
  return (
    <>
      <Switch>
        <Route exact path="/" component={Dashboards} />
        <Route path="/bi/:id" component={Bi} />
        <Route component={Page404} />
      </Switch>
    </>
  );
};

export default MainContainer;
