import { useLanguage } from '@/contexts/LanguageContext';
import { translations, TranslationKeys } from '@/lib/translations';

export const useTranslations = () => {
  const { language } = useLanguage();
  
  const t = (key: TranslationKeys): string => {
    return translations[language][key] || translations.en[key] || key;
  };
  
  return { t };
};