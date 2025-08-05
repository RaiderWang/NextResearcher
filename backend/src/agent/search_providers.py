from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pydantic import BaseModel, Field


@dataclass
class SearchResult:
    """标准化搜索结果格式
    
    用于统一不同搜索提供商返回的数据格式，确保搜索结果的一致性和可互换性。
    """
    content: str
    """搜索得到的文本内容，经过格式化处理"""
    
    sources: List[Dict[str, str]]
    """搜索来源列表，每个来源包含 {"label": str, "url": str, "title": str}"""
    
    metadata: Optional[Dict[str, Any]] = None
    """额外的元数据信息，如搜索用时、置信度等"""


class SearchMetrics(BaseModel):
    """搜索性能指标"""
    search_time: float = Field(description="搜索耗时（秒）")
    result_count: int = Field(description="返回结果数量")
    api_calls_used: int = Field(default=1, description="使用的API调用次数")
    tokens_consumed: Optional[int] = Field(default=None, description="消耗的token数量")


class SearchRequest(BaseModel):
    """标准化搜索请求格式"""
    query: str = Field(description="搜索查询字符串")
    max_results: int = Field(default=10, description="最大返回结果数量")
    language: Optional[str] = Field(default="zh-CN", description="搜索语言偏好")
    region: Optional[str] = Field(default=None, description="搜索地区偏好")
    date_restrict: Optional[str] = Field(default=None, description="时间范围限制")
    safe_search: bool = Field(default=True, description="是否启用安全搜索")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="提供商特定的额外参数")


class BaseSearchProvider(ABC):
    """搜索提供商基类
    
    定义了所有搜索提供商必须实现的接口，确保不同提供商的互换性。
    """
    
    def __init__(self, config: Dict[str, Any]):
        """初始化搜索提供商
        
        Args:
            config: 提供商特定的配置参数
        """
        self.config = config
        self._validate_config()
    
    @abstractmethod
    async def search(self, request: SearchRequest) -> SearchResult:
        """执行搜索并返回标准化结果
        
        Args:
            request: 标准化的搜索请求
            
        Returns:
            SearchResult: 标准化的搜索结果
            
        Raises:
            SearchProviderError: 当搜索失败时抛出
        """
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """返回提供商名称
        
        Returns:
            str: 提供商的唯一标识符
        """
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """验证提供商配置是否有效
        
        Returns:
            bool: 配置是否有效
        """
        pass
    
    @abstractmethod
    def get_required_config_keys(self) -> List[str]:
        """获取必需的配置键列表
        
        Returns:
            List[str]: 必需的配置键名称列表
        """
        pass
    
    @abstractmethod
    def get_rate_limits(self) -> Dict[str, Any]:
        """获取API速率限制信息
        
        Returns:
            Dict[str, Any]: 包含速率限制信息的字典
        """
        pass
    
    def _validate_config(self):
        """验证配置的内部方法"""
        required_keys = self.get_required_config_keys()
        missing_keys = [key for key in required_keys if key not in self.config]
        if missing_keys:
            raise SearchProviderConfigError(
                f"Missing required configuration keys for {self.get_provider_name()}: {missing_keys}"
            )
        
        if not self.validate_config():
            raise SearchProviderConfigError(
                f"Invalid configuration for {self.get_provider_name()}"
            )
    
    async def health_check(self) -> bool:
        """检查提供商服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
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
    
    def get_search_metrics(self) -> Optional[SearchMetrics]:
        """获取最近一次搜索的性能指标
        
        Returns:
            Optional[SearchMetrics]: 搜索指标，如果没有则返回None
        """
        return getattr(self, '_last_metrics', None)


class SearchProviderError(Exception):
    """搜索提供商基础异常类"""
    pass


class SearchProviderConfigError(SearchProviderError):
    """搜索提供商配置错误"""
    pass


class SearchProviderAPIError(SearchProviderError):
    """搜索提供商API错误"""
    pass


class SearchProviderTimeoutError(SearchProviderError):
    """搜索提供商超时错误"""
    pass


class SearchProviderRateLimitError(SearchProviderError):
    """搜索提供商速率限制错误"""
    pass 