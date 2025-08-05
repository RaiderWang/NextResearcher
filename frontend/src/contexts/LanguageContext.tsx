import React, { createContext, useContext, useEffect, useState } from 'react';

export type Language = 'en' | 'zh';

interface LanguageContextType {
  language: Language;
  toggleLanguage: () => void;
  setLanguage: (lang: Language) => void;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const useLanguage = () => {
  const context = useContext(LanguageContext);
  if (context === undefined) {
    throw new Error('useLanguage must be used within a LanguageProvider');
  }
  return context;
};

export const LanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [language, setLanguageState] = useState<Language>(() => {
    // 从localStorage获取保存的语言，默认为en
    const savedLanguage = localStorage.getItem('language') as Language;
    return savedLanguage || 'en';
  });

  useEffect(() => {
    // 保存到localStorage
    localStorage.setItem('language', language);
  }, [language]);

  const toggleLanguage = () => {
    setLanguageState(prevLanguage => prevLanguage === 'en' ? 'zh' : 'en');
  };

  const setLanguage = (lang: Language) => {
    setLanguageState(lang);
  };

  return (
    <LanguageContext.Provider value={{ language, toggleLanguage, setLanguage }}>
      {children}
    </LanguageContext.Provider>
  );
};