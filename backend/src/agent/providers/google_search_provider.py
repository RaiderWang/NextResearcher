import os
import time
import asyncio
from typing import List, Dict, Any, Optional
from google.genai import Client

from agent.search_providers import (
    BaseSearchProvider,
    SearchRequest,
    SearchResult,
    SearchMetrics,
    SearchProviderAPIError,
    SearchProviderTimeoutError,
    SearchProviderRateLimitError
)



class GoogleSearchProvider(BaseSearchProvider):
    """Google搜索提供商实现
    
    使用Google的GenAI客户端提供的搜索功能来执行网络搜索。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化Google搜索提供商
        
        Args:
            config: 配置字典，需要包含 'api_key' 字段
        """
        super().__init__(config)
        self.client = Client(api_key=config["api_key"])
        self.model = config.get("model", "gemini-2.0-flash")
        self._last_metrics: Optional[SearchMetrics] = None
    
    async def search(self, request: SearchRequest) -> SearchResult:
        """执行Google搜索
        
        Args:
            request: 搜索请求对象
            
        Returns:
            SearchResult: 标准化的搜索结果
        """
        start_time = time.time()
        
        try:
            # 构建搜索提示
            search_prompt = self._build_search_prompt(request)
            
            # 执行搜索 (使用异步包装同步调用)
            response = await asyncio.to_thread(
                self.client.models.generate_content,
                model=self.model,
                contents=search_prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "temperature": 0,
                }
            )
            
            # 处理搜索结果 (使用异步包装)
            search_result = await asyncio.to_thread(self._process_response, response, request)
            
            # 记录性能指标
            end_time = time.time()
            self._last_metrics = SearchMetrics(
                search_time=end_time - start_time,
                result_count=len(search_result.sources),
                api_calls_used=1,
                tokens_consumed=getattr(response, 'token_count', None)
            )
            
            return search_result
            
        except Exception as e:
            # 根据错误类型抛出相应的异常
            if "timeout" in str(e).lower():
                raise SearchProviderTimeoutError(f"Google搜索超时: {e}")
            elif "rate limit" in str(e).lower() or "quota" in str(e).lower():
                raise SearchProviderRateLimitError(f"Google搜索速率限制: {e}")
            else:
                raise SearchProviderAPIError(f"Google搜索API错误: {e}")
    
    def _build_search_prompt(self, request: SearchRequest) -> str:
        """构建搜索提示
        
        Args:
            request: 搜索请求
            
        Returns:
            str: 构建的搜索提示
        """
        prompt = f"""
请基于以下查询进行网络搜索，并提供详细的信息总结：

查询: {request.query}

要求:
- 搜索相关的最新信息
- 提供详细的内容总结
- 包含可靠的来源链接
- 最多返回 {request.max_results} 个相关结果
        """
        
        if request.language:
            prompt += f"\n- 优先搜索 {request.language} 语言的内容"
            
        if request.date_restrict:
            prompt += f"\n- 时间范围限制: {request.date_restrict}"
            
        return prompt.strip()
    
    def _clean_fake_citations(self, text: str) -> str:
        """清理Gemini API自动生成的假引用链接
        
        Args:
            text: 包含假链接的原始文本
            
        Returns:
            str: 清理后的文本
        """
        import re
        
        # 移除所有形如 [label](https://vertexaisearch.cloud.google.com/...) 的假链接
        # 但保留label文本
        pattern = r'\[([^\]]+)\]\(https://vertexaisearch\.cloud\.google\.com/[^)]+\)'
        cleaned_text = re.sub(pattern, r'\1', text)
        
        # 移除单独的vertexaisearch链接（没有label的情况）
        pattern2 = r'https://vertexaisearch\.cloud\.google\.com/[^\s\])]+'
        cleaned_text = re.sub(pattern2, '', cleaned_text)
        
        return cleaned_text.strip()
    
    def _process_response(self, response, request: SearchRequest) -> SearchResult:
        """处理Google搜索响应
        
        Args:
            response: Google GenAI响应对象
            request: 原始搜索请求
            
        Returns:
            SearchResult: 标准化的搜索结果
        """
        try:
            # 直接从grounding chunks获取真实URL，不使用resolve_urls转换
            grounding_chunks = response.candidates[0].grounding_metadata.grounding_chunks
            
            # 构建来源列表 - 直接使用真实URL
            sources = []
            seen_urls = set()
            
            for i, chunk in enumerate(grounding_chunks):
                if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri'):
                    real_url = chunk.web.uri  # 保留原始的vertexaisearch链接，这些会重定向到真实网站
                    title = getattr(chunk.web, 'title', f"搜索结果{i+1}")
                    
                    print(f"🔍 处理grounding chunk {i+1}: [{title}] -> {real_url[:50]}...")
                    
                    # 避免重复URL
                    if real_url not in seen_urls:
                        sources.append({
                            "label": f"来源{i+1}",
                            "url": real_url,  # 使用原始URL（Google的重定向链接）
                            "title": title
                        })
                        seen_urls.add(real_url)
                    else:
                        print(f"⏭️ 跳过重复URL")
            
            # 限制结果数量
            sources = sources[:request.max_results]
            
            # 清理Gemini API自动生成的假链接
            clean_content = self._clean_fake_citations(response.text)
            
            return SearchResult(
                content=clean_content,
                sources=sources,
                metadata={
                    "provider": "google",
                    "model": self.model,
                    "query": request.query,
                    "original_response_text": response.text,
                    "total_sources_found": len(sources)
                }
            )
            
        except Exception as e:
            # 如果处理失败，返回基础结果
            return SearchResult(
                content=response.text if hasattr(response, 'text') else "搜索失败",
                sources=[],
                metadata={
                    "provider": "google",
                    "model": self.model,
                    "query": request.query,
                    "error": str(e)
                }
            )
    
    def get_provider_name(self) -> str:
        """返回提供商名称"""
        return "google"
    
    def validate_config(self) -> bool:
        """验证配置是否有效"""
        try:
            api_key = self.config.get("api_key")
            if not api_key:
                return False
                
            # 验证API密钥格式（基本检查）
            if not isinstance(api_key, str) or len(api_key) < 10:
                return False
                
            return True
        except Exception:
            return False
    
    def get_required_config_keys(self) -> List[str]:
        """获取必需的配置键列表"""
        return ["api_key"]
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """获取API速率限制信息"""
        return {
            "requests_per_minute": 60,  # Google GenAI的典型限制
            "requests_per_day": 1500,
            "concurrent_requests": 5,
            "notes": "实际限制可能因API计划而异"
        }
    
    def get_supported_features(self) -> List[str]:
        """获取支持的搜索功能"""
        return [
            "web_search",
            "real_time_results",
            "grounding_metadata",
            "citation_support",
            "multi_language",
            "safe_search"
        ] 