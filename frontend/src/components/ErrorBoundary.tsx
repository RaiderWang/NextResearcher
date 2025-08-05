import React from 'react';

interface ErrorBoundaryState {
  hasError: boolean;
  error?: Error;
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('渲染错误:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="flex flex-col items-center justify-center h-full p-4">
          <h2 className="text-xl font-bold text-destructive mb-4">页面渲染出错</h2>
          <details className="bg-muted p-4 rounded max-w-2xl">
            <summary className="cursor-pointer mb-2">错误详情</summary>
            <pre className="text-sm overflow-auto">
              {this.state.error?.stack || this.state.error?.message}
            </pre>
          </details>
          <button 
            className="mt-4 px-4 py-2 bg-primary text-primary-foreground rounded"
            onClick={() => window.location.reload()}
          >
            重新加载
          </button>
        </div>
      );
    }

    return this.props.children;
  }
} 