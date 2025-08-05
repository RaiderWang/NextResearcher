import { Button } from "@/components/ui/button";
import { useLanguage } from "@/contexts/LanguageContext";
import { useTranslations } from "@/hooks/useTranslations";
import { Languages } from "lucide-react";

export const LanguageToggle = () => {
  const { language, toggleLanguage } = useLanguage();
  const { t } = useTranslations();

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={toggleLanguage}
      className="h-9 w-9 p-0"
      title={language === 'en' ? t('switchToChinese') : t('switchToEnglish')}
    >
      <Languages className="h-4 w-4" />
    </Button>
  );
};