import React, {FC, useState, useEffect, useContext, useReducer} from 'react';
import {RouteComponentProps} from 'react-router-dom';
import Dashboard from '../Dashboard';
import Spinner from '../../components/common/Spinner';
import {DASH_ACTIONS} from '../../utils/Consts';
import {Dashboard as DashboardType} from '../../types';
import {dashboardReducer} from '../../utils/reducers/dashboardReducer';
import {queryContext} from '../../contexts/QueryContext';

type DashboardPageProps = RouteComponentProps<{id?: string}>;

const Bi: FC<DashboardPageProps> = ({match}) => {
  const {getDashBoard, saveDashboard} = useContext(queryContext);
  const [loading, setLoading] = useState<boolean>(true);
  const [dashboard, setDashboard] = useReducer(dashboardReducer, null);
  const id = Number(match.params.id);

  useEffect(() => {
    getDashBoard(id).then((dashboard: DashboardType) => {
      if (dashboard) {
        setDashboard({type: DASH_ACTIONS.UPDATE, payload: dashboard});
        setLoading(false);
      }
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  // store dashboard on change
  useEffect(() => {
    if (dashboard) {
      // make changes persistent
      saveDashboard(JSON.stringify(dashboard), id);
    }

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dashboard]);

  if (loading) {
    return (
      <div className="loading-container">
        <Spinner />
      </div>
    );
  }

  return <Dashboard dashboard={dashboard} setDashboard={setDashboard} />;
};

export default Bi;
