export type Language = 'en' | 'zh';

export const translations = {
  en: {
    // App.tsx
    loadingConfig: 'Loading Configuration...',
    gettingSettings: 'Getting default settings',
    error: 'Error',
    retry: 'Retry',
    
    // WelcomeScreen.tsx
    welcome: 'Welcome',
    NextResearcher: 'Next Researcher',
    welcomeMessage: 'What can I do for you today?',
    poweredBy: 'Built with Google Gemini and LangChain LangGraph',
    
    // InputForm.tsx
    placeholder: 'Please enter your question...',
    search: 'Search',
    effort: 'Effort',
    effortLow: 'Low',
    effortMedium: 'Medium',
    effortHigh: 'High',
    provider: 'Provider',
    model: 'Model',
    searchProvider: 'Search',
    newSearch: 'New Search',
    loading: 'Loading...',
    
    // ChatMessagesView.tsx
    copied: 'Copied',
    copy: 'Copy',
    processing: 'Processing...',
    messageContentLength: 'Message content length',
    type: 'Type',
    
    // ActivityTimeline.tsx
    researchProcess: 'Research Process',
    searching: 'Searching...',
    noActivity: 'No activity records',
    timelineWillUpdate: 'Timeline will update during processing',
    
    // ThemeToggle.tsx
    switchToDark: 'Switch to dark mode',
    switchToLight: 'Switch to light mode',
    
    // LanguageToggle.tsx
    switchToEnglish: 'Switch to English',
    switchToChinese: 'Switch to Chinese',
    
    // Activity titles (for ActivityTimeline)
    generatingSearchQueries: 'Generating Search Queries',
    webResearch: 'Web Research',
    reflection: 'Reflection',
    finalizingAnswer: 'Finalizing Answer',
    analysisResults: 'Analysing Web Research Results',
    composingAnswer: 'Composing and presenting the final answer.',
    
    // Form placeholders and labels
    selectEffort: 'Effort',
    selectProvider: 'Provider',
    selectModel: 'Model',
    selectSearchProvider: 'Search',
  },
  zh: {
    // App.tsx
    loadingConfig: '加载配置中...',
    gettingSettings: '正在获取默认设置',
    error: '错误',
    retry: '重试',
    
    // WelcomeScreen.tsx
    welcome: '欢迎使用',
    NextResearcher: '问 道',
    welcomeMessage: '今天我可以为您做些什么？',
    poweredBy: '基于 Google Gemini 和 LangChain LangGraph 构建',
    
    // InputForm.tsx
    placeholder: '请输入您的问题...',
    search: '搜索',
    effort: '效果',
    effortLow: '低',
    effortMedium: '中',
    effortHigh: '高',
    provider: '提供商',
    model: '模型',
    searchProvider: '搜索',
    newSearch: '新搜索',
    loading: '加载中...',
    
    // ChatMessagesView.tsx
    copied: '已复制',
    copy: '复制',
    processing: '处理中...',
    messageContentLength: '消息内容长度',
    type: '类型',
    
    // ActivityTimeline.tsx
    researchProcess: '研究过程',
    searching: '搜索中...',
    noActivity: '暂无活动记录',
    timelineWillUpdate: '处理过程中时间线将更新',
    
    // ThemeToggle.tsx
    switchToDark: '切换到深色模式',
    switchToLight: '切换到浅色模式',
    
    // LanguageToggle.tsx
    switchToEnglish: '切换到英文',
    switchToChinese: '切换到中文',
    
    // Activity titles (for ActivityTimeline)
    generatingSearchQueries: '生成搜索查询',
    webResearch: '网络研究',
    reflection: '反思分析',
    finalizingAnswer: '最终答案',
    analysisResults: '分析网络研究结果',
    composingAnswer: '正在撰写并呈现最终答案。',
    
    // Form placeholders and labels
    selectEffort: '效果',
    selectProvider: '提供商',
    selectModel: '模型',
    selectSearchProvider: '搜索',
  },
} as const;

export type TranslationKeys = keyof typeof translations.en;