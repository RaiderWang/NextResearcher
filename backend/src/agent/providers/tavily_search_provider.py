import os
import time
import aiohttp
import asyncio
from typing import List, Dict, Any, Optional

from agent.search_providers import (
    BaseSearchProvider,
    SearchRequest,
    SearchResult,
    SearchMetrics,
    SearchProviderAPIError,
    SearchProviderTimeoutError,
    SearchProviderRateLimitError
)


class TavilySearchProvider(BaseSearchProvider):
    """Tavily搜索提供商实现
    
    使用Tavily的REST API提供网络搜索功能，专为AI应用优化。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化Tavily搜索提供商
        
        Args:
            config: 配置字典，需要包含 'api_key' 字段
        """
        super().__init__(config)
        self.api_key = config["api_key"]
        self.base_url = config.get("base_url", "https://api.tavily.com")
        self.timeout = config.get("timeout", 30.0)
        self._last_metrics: Optional[SearchMetrics] = None
    
    async def search(self, request: SearchRequest) -> SearchResult:
        """执行Tavily搜索
        
        Args:
            request: 搜索请求对象
            
        Returns:
            SearchResult: 标准化的搜索结果
        """
        start_time = time.time()
        
        try:
            # 构建搜索请求
            search_payload = self._build_search_payload(request)
            
            # 执行搜索
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/search",
                    json=search_payload,
                    headers={
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    },
                    timeout=aiohttp.ClientTimeout(total=self.timeout)
                ) as response:
                    
                    if response.status == 429:
                        raise SearchProviderRateLimitError("Tavily API请求频率限制")
                    elif response.status == 401:
                        raise SearchProviderAPIError("Tavily API认证失败，请检查API密钥")
                    elif response.status != 200:
                        error_text = await response.text()
                        raise SearchProviderAPIError(f"Tavily API错误 {response.status}: {error_text}")
                    
                    response_data = await response.json()
            
            # 处理搜索结果
            search_result = self._process_response(response_data, request)
            
            # 记录性能指标
            end_time = time.time()
            self._last_metrics = SearchMetrics(
                search_time=end_time - start_time,
                result_count=len(search_result.sources),
                api_calls_used=1
            )
            
            return search_result
            
        except asyncio.TimeoutError:
            raise SearchProviderTimeoutError("Tavily搜索请求超时")
        except aiohttp.ClientError as e:
            raise SearchProviderAPIError(f"Tavily网络请求错误: {e}")
        except Exception as e:
            if isinstance(e, (SearchProviderAPIError, SearchProviderTimeoutError, SearchProviderRateLimitError)):
                raise
            else:
                raise SearchProviderAPIError(f"Tavily搜索未知错误: {e}")
    
    def _build_search_payload(self, request: SearchRequest) -> Dict[str, Any]:
        """构建Tavily搜索请求载荷
        
        Args:
            request: 搜索请求
            
        Returns:
            Dict[str, Any]: API请求载荷
        """
        payload = {
            "query": request.query,
            "search_depth": "advanced",  # basic 或 advanced
            "include_answer": True,
            "include_raw_content": False,
            "max_results": min(request.max_results, 10),  # Tavily限制最多10个结果
            "include_domains": [],
            "exclude_domains": []
        }
        
        # 添加语言和地区配置
        if request.language:
            # 将语言代码转换为Tavily支持的格式
            language_map = {
                "zh-CN": "zh",
                "zh-TW": "zh",
                "en-US": "en",
                "en": "en",
                "ja": "ja",
                "ko": "ko",
                "fr": "fr",
                "de": "de",
                "es": "es"
            }
            tavily_lang = language_map.get(request.language, "en")
            payload["include_answer"] = True
        
        # 添加时间限制
        if request.date_restrict:
            # Tavily可能支持时间过滤，这里添加到额外参数中
            payload["days"] = self._parse_date_restrict(request.date_restrict)
        
        # 添加安全搜索
        if not request.safe_search:
            payload["search_depth"] = "basic"  # 基础搜索可能包含更多内容
        
        # 合并额外参数
        if request.additional_params:
            payload.update(request.additional_params)
        
        return payload
    
    def _parse_date_restrict(self, date_restrict: str) -> Optional[int]:
        """解析时间限制字符串
        
        Args:
            date_restrict: 时间限制字符串，如 "7d", "1m", "1y"
            
        Returns:
            Optional[int]: 天数限制
        """
        if not date_restrict:
            return None
            
        try:
            if date_restrict.endswith('d'):
                return int(date_restrict[:-1])
            elif date_restrict.endswith('w'):
                return int(date_restrict[:-1]) * 7
            elif date_restrict.endswith('m'):
                return int(date_restrict[:-1]) * 30
            elif date_restrict.endswith('y'):
                return int(date_restrict[:-1]) * 365
        except ValueError:
            pass
        
        return None
    
    def _process_response(self, response_data: Dict[str, Any], request: SearchRequest) -> SearchResult:
        """处理Tavily搜索响应
        
        Args:
            response_data: Tavily API响应数据
            request: 原始搜索请求
            
        Returns:
            SearchResult: 标准化的搜索结果
        """
        try:
            # 提取答案内容
            answer = response_data.get("answer", "")
            
            # 提取搜索结果
            results = response_data.get("results", [])
            
            # 构建内容字符串
            content_parts = []
            if answer:
                content_parts.append(f"概要答案：{answer}")
            
            # 添加详细结果
            if results:
                content_parts.append("\n详细搜索结果：")
                for i, result in enumerate(results[:request.max_results], 1):
                    title = result.get("title", "无标题")
                    content = result.get("content", "")
                    if content:
                        content_parts.append(f"\n{i}. {title}")
                        content_parts.append(f"   {content[:200]}...")
            
            content = "\n".join(content_parts)
            
            # 构建来源列表
            sources = []
            for i, result in enumerate(results[:request.max_results]):
                source = {
                    "label": f"来源{i+1}",
                    "url": result.get("url", ""),
                    "title": result.get("title", f"搜索结果{i+1}")
                }
                sources.append(source)
            
            return SearchResult(
                content=content,
                sources=sources,
                metadata={
                    "provider": "tavily",
                    "query": request.query,
                    "answer": answer,
                    "total_results": len(results),
                    "search_depth": "advanced",
                    "response_data": response_data
                }
            )
            
        except Exception as e:
            # 如果处理失败，返回基础结果
            return SearchResult(
                content=str(response_data.get("answer", "搜索失败")),
                sources=[],
                metadata={
                    "provider": "tavily",
                    "query": request.query,
                    "error": str(e),
                    "raw_response": response_data
                }
            )
    
    def get_provider_name(self) -> str:
        """返回提供商名称"""
        return "tavily"
    
    def validate_config(self) -> bool:
        """验证配置是否有效"""
        try:
            api_key = self.config.get("api_key")
            if not api_key:
                return False
                
            # 验证API密钥格式（Tavily API密钥通常以tvly-开头）
            if not isinstance(api_key, str) or len(api_key) < 20:
                return False
            
            # 可选：检查API密钥前缀
            # if not api_key.startswith("tvly-"):
            #     return False
                
            return True
        except Exception:
            return False
    
    def get_required_config_keys(self) -> List[str]:
        """获取必需的配置键列表"""
        return ["api_key"]
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """获取API速率限制信息"""
        return {
            "requests_per_minute": 100,  # Tavily的典型限制
            "requests_per_day": 1000,
            "concurrent_requests": 10,
            "max_results_per_request": 10,
            "notes": "实际限制根据订阅计划而异"
        }
    
    def get_supported_features(self) -> List[str]:
        """获取支持的搜索功能"""
        return [
            "web_search",
            "real_time_results",
            "ai_optimized_results",
            "answer_generation",
            "content_extraction",
            "multi_language",
            "safe_search",
            "domain_filtering"
        ]
    
    async def health_check(self) -> bool:
        """检查Tavily服务是否可用"""
        try:
            # 执行一个简单的测试搜索
            test_request = SearchRequest(
                query="test",
                max_results=1
            )
            await self.search(test_request)
            return True
        except Exception:
            return False 