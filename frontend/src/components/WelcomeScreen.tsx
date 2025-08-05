import { InputForm } from "./InputForm";
import { useTranslations } from "@/hooks/useTranslations";

interface WelcomeScreenProps {
  handleSubmit: (
    submittedInputValue: string,
    effort: string,
    model: string,
    searchProvider: string,
    llmProvider: string
  ) => void;
  onCancel: () => void;
  isLoading: boolean;
  formState: {
    effort: string;
    model: string;
    searchProvider: string;
    llmProvider: string;
  };
  updateFormState: (updates: Partial<{
    effort: string;
    model: string;
    searchProvider: string;
    llmProvider: string;
  }>) => void;
}

export const WelcomeScreen: React.FC<WelcomeScreenProps> = ({
  handleSubmit,
  onCancel,
  isLoading,
  formState,
  updateFormState,
}) => {
  const { t } = useTranslations();
  
  return (
  <div className="h-full flex flex-col items-center justify-center text-center px-4 flex-1 w-full max-w-3xl mx-auto gap-4">
    <div>
      <h1 className="text-5xl md:text-6xl font-semibold text-foreground mb-3">
      {t('NextResearcher')}
      </h1>
      <p className="text-xl md:text-2xl text-muted-foreground">
        {t('welcomeMessage')}
      </p>
    </div>
    <div className="w-full mt-4">
      <InputForm
        onSubmit={handleSubmit}
        isLoading={isLoading}
        onCancel={onCancel}
        hasHistory={false}
        formState={formState}
        updateFormState={updateFormState}
      />
    </div>
    <p className="text-xs text-muted-foreground">
      {/* {t('poweredBy')} */}
    </p>
  </div>
  );
};
