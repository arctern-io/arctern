import React, {FC, useState, createContext, ReactNode} from 'react';
import {namespace} from '../utils/Helpers';

export type UserAuth = {
  userId: string;
  token: string;
  expired: number;
};

interface IAuthContext {
  auth: UserAuth;
  setAuthStatus: (userAuth: UserAuth) => void;
  setUnauthStatus: () => void;
}

export const DEFAULT_USER_AUTH = {userId: 'guest', token: 'bi@zilliz.com', expired: 0};
export const authContext = createContext<IAuthContext>({
  auth: DEFAULT_USER_AUTH,
  setAuthStatus: () => {},
  setUnauthStatus: () => {},
});

export const getStoredUserAuth = (): UserAuth => {
  const auth = window.localStorage.getItem(namespace(['login'], 'userAuth'));
  if (auth) {
    return JSON.parse(auth);
  }
  return DEFAULT_USER_AUTH;
};

const useAuthHandler = (initialState: UserAuth) => {
  const [auth, setAuth] = useState(initialState);
  const setAuthStatus = (userAuth: UserAuth) => {
    window.localStorage.setItem(namespace(['login'], 'userAuth'), JSON.stringify(userAuth));
    setAuth(userAuth);
  };
  const setUnauthStatus = () => {
    window.localStorage.removeItem(namespace(['login'], 'userAuth'));
    setAuth(DEFAULT_USER_AUTH);
  };

  return {
    auth,
    setAuthStatus,
    setUnauthStatus,
  };
};

const {Provider} = authContext;
const AuthProvider: FC<{children: ReactNode}> = ({children}) => {
  const {auth, setAuthStatus, setUnauthStatus} = useAuthHandler(getStoredUserAuth());

  return (
    <Provider
      value={{
        auth,
        setAuthStatus,
        setUnauthStatus,
      }}
    >
      {children}
    </Provider>
  );
};
export default AuthProvider;
