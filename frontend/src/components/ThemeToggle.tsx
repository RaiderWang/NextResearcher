import { Button } from "@/components/ui/button";
import { useTheme } from "@/contexts/ThemeContext";
import { useTranslations } from "@/hooks/useTranslations";
import { Sun, Moon } from "lucide-react";

export const ThemeToggle = () => {
  const { theme, toggleTheme } = useTheme();
  const { t } = useTranslations();

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={toggleTheme}
      className="h-9 w-9 p-0"
      title={theme === 'light' ? t('switchToDark') : t('switchToLight')}
    >
      {theme === 'light' ? (
        <Moon className="h-4 w-4" />
      ) : (
        <Sun className="h-4 w-4" />
      )}
    </Button>
  );
}; 