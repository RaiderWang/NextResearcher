import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { SquarePen, Brain, Send, StopCircle, Cpu, Search, Settings } from "lucide-react";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useTranslations } from "@/hooks/useTranslations";

// LLM Provider and Model types
interface LLMModel {
  id: string;
  name: string;
  description: string;
  context_length: number;
  supports_structured_output: boolean;
}

interface LLMProvider {
  name: string;
  display_name: string;
  available: boolean;
  models: LLMModel[];
  reason?: string;
}

interface LLMProvidersResponse {
  providers: Record<string, LLMProvider>;
  default_provider: string;
}

// Updated InputFormProps
interface InputFormProps {
  onSubmit: (inputValue: string, effort: string, model: string, searchProvider: string, llmProvider: string) => void;
  onCancel: () => void;
  isLoading: boolean;
  hasHistory: boolean;
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

export const InputForm: React.FC<InputFormProps> = ({
  onSubmit,
  onCancel,
  isLoading,
  hasHistory,
  formState,
  updateFormState,
}) => {
  const { t } = useTranslations();
  const [internalInputValue, setInternalInputValue] = useState("");
  const [llmProviders, setLlmProviders] = useState<LLMProvidersResponse | null>(null);
  const [loadingProviders, setLoadingProviders] = useState(true);
  const { effort, model, searchProvider, llmProvider } = formState;

  // 获取LLM提供商列表
  useEffect(() => {
    const fetchProviders = async () => {
      try {
        const response = await fetch(
          import.meta.env.DEV
            ? "http://localhost:2024/api/llm-providers"
            : "http://localhost:8123/api/llm-providers"
        );
        if (response.ok) {
          const data: LLMProvidersResponse = await response.json();
          setLlmProviders(data);
          
          // 验证当前选择的provider和model是否有效
          if (llmProvider && data.providers[llmProvider]) {
            const currentProvider = data.providers[llmProvider];
            if (currentProvider.available) {
              // 检查当前选择的model是否在提供商的模型列表中
              const modelExists = currentProvider.models.some(m => m.id === model);
              if (!modelExists && currentProvider.models.length > 0) {
                // 如果当前model不存在，选择该提供商的第一个model
                updateFormState({ model: currentProvider.models[0].id });
              }
            } else {
              // 如果当前provider不可用，回退到默认provider
              if (data.default_provider && data.providers[data.default_provider]) {
                const defaultProvider = data.providers[data.default_provider];
                if (defaultProvider.available && defaultProvider.models.length > 0) {
                  updateFormState({
                    llmProvider: data.default_provider,
                    model: defaultProvider.models[0].id
                  });
                }
              }
            }
          }
        }
      } catch (error) {
        console.error('Failed to fetch LLM providers:', error);
      } finally {
        setLoadingProviders(false);
      }
    };

    // 只有当有llmProvider时才获取providers，避免与默认配置加载冲突
    if (llmProvider) {
      fetchProviders();
    }
  }, [llmProvider, model, updateFormState]);

  // 当提供商改变时，更新模型选择
  const handleProviderChange = (newProvider: string) => {
    const provider = llmProviders?.providers[newProvider];
    if (provider && provider.available && provider.models.length > 0) {
      updateFormState({
        llmProvider: newProvider,
        model: provider.models[0].id
      });
    }
  };

  // 获取当前提供商的模型列表
  const getCurrentProviderModels = (): LLMModel[] => {
    if (!llmProviders || !llmProvider) return [];
    const provider = llmProviders.providers[llmProvider];
    return provider?.models || [];
  };

  // 获取可用的提供商列表
  const getAvailableProviders = (): LLMProvider[] => {
    if (!llmProviders) return [];
    return Object.values(llmProviders.providers).filter(provider => provider.available);
  };

  const handleInternalSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!internalInputValue.trim()) return;
    onSubmit(internalInputValue, effort, model, searchProvider, llmProvider);
    setInternalInputValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Submit with Ctrl+Enter (Windows/Linux) or Cmd+Enter (Mac)
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleInternalSubmit();
    }
  };

  const isSubmitDisabled = !internalInputValue.trim() || isLoading;

  return (
    <form
      onSubmit={handleInternalSubmit}
      className={`flex flex-col gap-2 p-3 pb-4`}
    >
      <div
        className={`flex flex-row items-center justify-between text-foreground rounded-3xl rounded-bl-sm ${
          hasHistory ? "rounded-br-sm" : ""
        } break-words min-h-7 bg-card border border-border px-4 pt-3 `}
      >
        <Textarea
          value={internalInputValue}
          onChange={(e) => setInternalInputValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={t('placeholder')}
          className={`w-full text-foreground placeholder-muted-foreground resize-none border-0 focus:outline-none focus:ring-0 outline-none focus-visible:ring-0 shadow-none !bg-transparent !border-none !shadow-none focus-visible:!ring-0 focus-visible:!border-transparent dark:!bg-transparent
                        md:text-base  min-h-[56px] max-h-[200px]`}
          rows={1}
        />
        <div className="-mt-3">
          {isLoading ? (
            <Button
              type="button"
              variant="ghost"
              size="icon"
              className="text-red-500 hover:text-red-400 hover:bg-red-500/10 p-2 cursor-pointer rounded-full transition-all duration-200"
              onClick={onCancel}
            >
              <StopCircle className="h-5 w-5" />
            </Button>
          ) : (
            <Button
              type="submit"
              variant="ghost"
              className={`${
                isSubmitDisabled
                  ? "text-muted-foreground"
                  : "text-primary hover:text-primary/80 hover:bg-primary/10"
              } p-2 cursor-pointer rounded-full transition-all duration-200 text-base`}
              disabled={isSubmitDisabled}
            >
              {t('search')}
              <Send className="h-5 w-5" />
            </Button>
          )}
        </div>
      </div>
      <div className="flex items-center justify-between">
        <div className="flex flex-row gap-2 flex-wrap">
          <div className="flex flex-row gap-2 bg-card border border-border text-card-foreground focus:ring-ring rounded-xl rounded-t-sm pl-2 max-w-[100%] sm:max-w-[90%]">
            <div className="flex flex-row items-center text-sm">
              <Brain className="h-4 w-4 mr-2" />
              {t('effort')}
            </div>
            <Select value={effort} onValueChange={(value) => updateFormState({ effort: value })}>
              <SelectTrigger className="w-[120px] bg-transparent border-none cursor-pointer">
                <SelectValue placeholder={t('selectEffort')} />
              </SelectTrigger>
              <SelectContent className="bg-card border-border text-card-foreground cursor-pointer">
                <SelectItem
                  value="low"
                  className="hover:bg-accent focus:bg-accent cursor-pointer"
                >
                  {t('effortLow')}
                </SelectItem>
                <SelectItem
                  value="medium"
                  className="hover:bg-accent focus:bg-accent cursor-pointer"
                >
                  {t('effortMedium')}
                </SelectItem>
                <SelectItem
                  value="high"
                  className="hover:bg-accent focus:bg-accent cursor-pointer"
                >
                  {t('effortHigh')}
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
          
          {/* LLM提供商选择器 */}
          <div className="flex flex-row gap-2 bg-card border border-border text-card-foreground focus:ring-ring rounded-xl rounded-t-sm pl-2 max-w-[100%] sm:max-w-[90%]">
            <div className="flex flex-row items-center text-sm">
              <Settings className="h-4 w-4 mr-2" />
              {t('provider')}
            </div>
            <Select
              value={llmProvider}
              onValueChange={handleProviderChange}
              disabled={loadingProviders}
            >
              <SelectTrigger className="w-[140px] bg-transparent border-none cursor-pointer">
                <SelectValue placeholder={loadingProviders ? t('loading') : t('selectProvider')} />
              </SelectTrigger>
              <SelectContent className="bg-card border-border text-card-foreground cursor-pointer">
                {getAvailableProviders().map((provider) => (
                  <SelectItem
                    key={provider.name}
                    value={provider.name}
                    className="hover:bg-accent focus:bg-accent cursor-pointer"
                  >
                    <div className="flex items-center">
                      <Settings className="h-4 w-4 mr-2 text-blue-500" />
                      {provider.display_name}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          {/* 模型选择器 */}
          <div className="flex flex-row gap-2 bg-card border border-border text-card-foreground focus:ring-ring rounded-xl rounded-t-sm pl-2 max-w-[100%] sm:max-w-[90%]">
            <div className="flex flex-row items-center text-sm ml-2">
              <Cpu className="h-4 w-4 mr-2" />
              {t('model')}
            </div>
            <Select
              value={model}
              onValueChange={(value) => updateFormState({ model: value })}
              disabled={loadingProviders || getCurrentProviderModels().length === 0}
            >
              <SelectTrigger className="w-[150px] bg-transparent border-none cursor-pointer">
                <SelectValue placeholder={loadingProviders ? t('loading') : t('selectModel')} />
              </SelectTrigger>
              <SelectContent className="bg-card border-border text-card-foreground cursor-pointer">
                {getCurrentProviderModels().map((modelItem) => (
                  <SelectItem
                    key={modelItem.id}
                    value={modelItem.id}
                    className="hover:bg-accent focus:bg-accent cursor-pointer"
                  >
                    <div className="flex items-center">
                      <Cpu className="h-4 w-4 mr-2 text-purple-400" />
                      {modelItem.name}
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
          <div className="flex flex-row gap-2 bg-card border border-border text-card-foreground focus:ring-ring rounded-xl rounded-t-sm pl-2  max-w-[100%] sm:max-w-[90%]">
            <div className="flex flex-row items-center text-sm ml-2">
              <Search className="h-4 w-4 mr-2" />
              {t('searchProvider')}
            </div>
            <Select value={searchProvider} onValueChange={(value) => updateFormState({ searchProvider: value })}>
              <SelectTrigger className="w-[120px] bg-transparent border-none cursor-pointer">
                <SelectValue placeholder={t('selectSearchProvider')} />
              </SelectTrigger>
              <SelectContent className="bg-card border-border text-card-foreground cursor-pointer">
                <SelectItem
                  value="google"
                  className="hover:bg-accent focus:bg-accent cursor-pointer"
                >
                  <div className="flex items-center">
                    <Search className="h-4 w-4 mr-2 text-blue-500" /> Google
                  </div>
                </SelectItem>
                <SelectItem
                  value="tavily"
                  className="hover:bg-accent focus:bg-accent cursor-pointer"
                >
                  <div className="flex items-center">
                    <Search className="h-4 w-4 mr-2 text-green-500" /> Tavily
                  </div>
                </SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        {hasHistory && (
          <Button
            className="bg-card border border-border text-card-foreground cursor-pointer rounded-xl rounded-t-sm pl-2 "
            variant="secondary"
            onClick={() => window.location.reload()}
          >
            <SquarePen size={16} />
            {t('newSearch')}
          </Button>
        )}
      </div>
    </form>
  );
};
