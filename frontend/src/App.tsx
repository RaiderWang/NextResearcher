import { useStream } from "@langchain/langgraph-sdk/react";
import type { Message } from "@langchain/langgraph-sdk";
import { useState, useEffect, useRef, useCallback } from "react";
import { ProcessedEvent } from "@/components/ActivityTimeline";
import { WelcomeScreen } from "@/components/WelcomeScreen";
import { ChatMessagesView } from "@/components/ChatMessagesView";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/ThemeToggle";
import { LanguageToggle } from "@/components/LanguageToggle";
import { ErrorBoundary } from "@/components/ErrorBoundary";
import { useTranslations } from "@/hooks/useTranslations";

export default function App() {
  const { t } = useTranslations();
  const [processedEventsTimeline, setProcessedEventsTimeline] = useState<
    ProcessedEvent[]
  >([]);
  const [historicalActivities, setHistoricalActivities] = useState<
    Record<string, ProcessedEvent[]>
  >({});
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const hasFinalizeEventOccurredRef = useRef(false);
  const [error, setError] = useState<string | null>(null);
  
  // 提升表单状态到App级别，防止组件切换时重置
  const [formState, setFormState] = useState({
    effort: "",
    model: "",
    searchProvider: "",
    llmProvider: ""
  });
  const [configLoaded, setConfigLoaded] = useState(false);
  const thread = useStream<{
    messages: Message[];
    initial_search_query_count: number;
    max_research_loops: number;
    reasoning_model: string;
    search_provider: string;
    llm_provider: string;
  }>({
    apiUrl: import.meta.env.DEV
      ? "http://localhost:2024"
      : "http://localhost:8123",
    assistantId: "agent",
    messagesKey: "messages",
    onUpdateEvent: (event: any) => {
      try {
        let processedEvent: ProcessedEvent | null = null;
        if (event.generate_query) {
          // 添加更安全的错误处理
          const searchQueries = event.generate_query?.search_query;
          const queriesText = Array.isArray(searchQueries) && searchQueries.length > 0 
            ? searchQueries.join(", ") 
            : "Generating search queries...";
          processedEvent = {
            title: t('generatingSearchQueries'),
            data: queriesText,
          };
        } else if (event.web_research) {
          const sources = event.web_research.sources_gathered || [];
          const numSources = sources.length;
          const uniqueLabels = [
            ...new Set(sources.map((s: any) => s.label).filter(Boolean)),
          ];
          const exampleLabels = uniqueLabels.slice(0, 3).join(", ");
          processedEvent = {
            title: t('webResearch'),
            data: `Gathered ${numSources} sources. Related to: ${
              exampleLabels || "N/A"
            }.`,
          };
        } else if (event.reflection) {
          processedEvent = {
            title: t('reflection'),
            data: t('analysisResults'),
          };
        } else if (event.finalize_answer) {
          processedEvent = {
            title: t('finalizingAnswer'),
            data: t('composingAnswer'),
          };
          hasFinalizeEventOccurredRef.current = true;
        }
        if (processedEvent) {
          setProcessedEventsTimeline((prevEvents) => [
            ...prevEvents,
            processedEvent!,
          ]);
        }
      } catch (error) {
        console.error("Event processing error:", error, "Event:", event);
        // 即使发生错误也要提供用户反馈
        setProcessedEventsTimeline((prevEvents) => [
          ...prevEvents,
          {
            title: "Processing Event",
            data: "Event processing in progress...",
          },
        ]);
      }
    },
    onError: (error: any) => {
      setError(error.message);
    },
  });

  useEffect(() => {
    if (scrollAreaRef.current) {
      const scrollViewport = scrollAreaRef.current.querySelector(
        "[data-radix-scroll-area-viewport]"
      );
      if (scrollViewport) {
        scrollViewport.scrollTop = scrollViewport.scrollHeight;
      }
    }
  }, [thread.messages]);

  useEffect(() => {
    // 添加调试日志
    console.log("状态更新:", {
      messagesCount: thread.messages.length,
      isLoading: thread.isLoading,
      lastMessage: thread.messages.length > 0 ? {
        type: thread.messages[thread.messages.length - 1].type,
        contentLength: typeof thread.messages[thread.messages.length - 1].content === "string" 
          ? thread.messages[thread.messages.length - 1].content.length 
          : 0,
        id: thread.messages[thread.messages.length - 1].id
      } : null
    });
    
    if (
      hasFinalizeEventOccurredRef.current &&
      !thread.isLoading &&
      thread.messages.length > 0
    ) {
      const lastMessage = thread.messages[thread.messages.length - 1];
      if (lastMessage && lastMessage.type === "ai" && lastMessage.id) {
        setHistoricalActivities((prev) => ({
          ...prev,
          [lastMessage.id!]: [...processedEventsTimeline],
        }));
      }
      hasFinalizeEventOccurredRef.current = false;
    }
  }, [thread.messages, thread.isLoading, processedEventsTimeline]);

  // 从后端加载默认配置
  useEffect(() => {
    const loadDefaultConfig = async () => {
      try {
        const response = await fetch(
          import.meta.env.DEV
            ? "http://localhost:2024/api/default-config"
            : "http://localhost:8123/api/default-config"
        );
        if (response.ok) {
          const defaultConfig = await response.json();
          setFormState({
            effort: defaultConfig.effort,
            model: defaultConfig.model,
            searchProvider: defaultConfig.search_provider,
            llmProvider: defaultConfig.llm_provider
          });
        } else {
          // 如果API失败，使用硬编码默认值
          setFormState({
            effort: "medium",
            model: "gemini-2.5-flash",
            searchProvider: "google",
            llmProvider: "GOOGLE_GEMINI"
          });
        }
      } catch (error) {
        console.error("Failed to load default config:", error);
        // 如果请求失败，使用硬编码默认值
        setFormState({
          effort: "medium",
          model: "gemini-2.5-flash",
          searchProvider: "google",
          llmProvider: "GOOGLE_GEMINI"
        });
      } finally {
        setConfigLoaded(true);
      }
    };

    loadDefaultConfig();
  }, []);

  const updateFormState = useCallback((updates: Partial<typeof formState>) => {
    setFormState(prev => ({ ...prev, ...updates }));
  }, []);

  const handleSubmit = useCallback(
    (submittedInputValue: string, effort: string, model: string, searchProvider: string, llmProvider: string) => {
      if (!submittedInputValue.trim()) return;
      
      // 更新表单状态，保持用户选择
      updateFormState({ effort, model, searchProvider, llmProvider });
      
      setProcessedEventsTimeline([]);
      hasFinalizeEventOccurredRef.current = false;

      // convert effort to, initial_search_query_count and max_research_loops
      // low means max 1 loop and 1 query
      // medium means max 3 loops and 3 queries
      // high means max 10 loops and 5 queries
      let initial_search_query_count = 0;
      let max_research_loops = 0;
      switch (effort) {
        case "low":
          initial_search_query_count = 1;
          max_research_loops = 1;
          break;
        case "medium":
          initial_search_query_count = 3;
          max_research_loops = 3;
          break;
        case "high":
          initial_search_query_count = 5;
          max_research_loops = 10;
          break;
      }

      const newMessages: Message[] = [
        ...(thread.messages || []),
        {
          type: "human",
          content: submittedInputValue,
          id: Date.now().toString(),
        },
      ];
      thread.submit({
        messages: newMessages,
        initial_search_query_count: initial_search_query_count,
        max_research_loops: max_research_loops,
        reasoning_model: model,
        search_provider: searchProvider,
        llm_provider: llmProvider,
      });
    },
    [thread, updateFormState]
  );

  const handleCancel = useCallback(() => {
    thread.stop();
    window.location.reload();
  }, [thread]);

  return (
    <div className="flex h-screen bg-background text-foreground font-sans antialiased">
      {/* 主题和语言切换按钮 - 固定在右上角 */}
      <div className="absolute top-4 right-4 z-50 flex items-center gap-2">
        <LanguageToggle />
        <ThemeToggle />
      </div>
      
      <main className="h-full w-full max-w-4xl mx-auto">
          {!configLoaded ? (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="text-center">
                <h2 className="text-2xl font-semibold text-foreground mb-3">
                  {t('loadingConfig')}
                </h2>
                <p className="text-muted-foreground">
                  {t('gettingSettings')}
                </p>
              </div>
            </div>
          ) : thread.messages.length === 0 ? (
            <WelcomeScreen
              handleSubmit={handleSubmit}
              isLoading={thread.isLoading}
              onCancel={handleCancel}
              formState={formState}
              updateFormState={updateFormState}
            />
          ) : error ? (
            <div className="flex flex-col items-center justify-center h-full">
              <div className="flex flex-col items-center justify-center gap-4">
                <h1 className="text-2xl text-destructive font-bold">{t('error')}</h1>
                <p className="text-destructive">{JSON.stringify(error)}</p>

                <Button
                  variant="destructive"
                  onClick={() => window.location.reload()}
                >
                  {t('retry')}
                </Button>
              </div>
            </div>
          ) : (
            <ErrorBoundary>
              <ChatMessagesView
                messages={thread.messages}
                isLoading={thread.isLoading}
                scrollAreaRef={scrollAreaRef}
                onSubmit={handleSubmit}
                onCancel={handleCancel}
                liveActivityEvents={processedEventsTimeline}
                historicalActivities={historicalActivities}
                formState={formState}
                updateFormState={updateFormState}
              />
            </ErrorBoundary>
          )}
      </main>
    </div>
  );
}
