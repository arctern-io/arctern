import React, {FC, useState, useEffect} from 'react';
import {timeFormatDefaultLocale} from 'd3';

interface II18nContextInterface {
  language: string;
  nls: any;
  setI18n: (language: string) => void;
}

export const I18nContext = React.createContext<II18nContextInterface>({
  language: 'en',
  nls: {},
  setI18n: () => {},
});

const {Provider} = I18nContext;
const languageEnabled = ['en', 'en-US', 'zh-CN'];
const I18nProvider: FC<{children: React.ReactNode}> = ({children}) => {
  let lan = navigator.language;

  if (lan === 'en') {
    lan = 'en-US';
  }

  lan = languageEnabled.indexOf(lan) !== -1 ? lan : 'en-US';

  let [language, setLanguage] = useState<string>(lan);
  let [nls, setNls] = useState<any>(null);

  useEffect(() => {
    import(`../i18n/${language}`).then(nls => {
      setNls(nls.default);
      // update d3 locale
      timeFormatDefaultLocale(nls.default.time_locale);
    });
  }, [language]);

  return <Provider value={{language, nls, setI18n: setLanguage}}>{children}</Provider>;
};

export default I18nProvider;
