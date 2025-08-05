"""LLM提供商基础接口和抽象类

定义了所有LLM提供商必须实现的接口，确保不同提供商的互换性。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Type
import asyncio
import time
from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from .llm_types import (
    LLMRequest,
    LLMResponse,
    LLMProviderConfig,
    LLMProviderType,
    LLMModel,
    LLMProviderError,
    LLMProviderConfigError,
    LLMTaskType
)


class BaseLLMProvider(ABC):
    """LLM提供商基类
    
    定义了所有LLM提供商必须实现的接口，确保不同提供商的互换性。
    """
    
    def __init__(self, config: LLMProviderConfig):
        """初始化LLM提供商
        
        Args:
            config: 提供商特定的配置参数
        """
        self.config = config
        self._validate_config()
        self._models_cache: Optional[List[LLMModel]] = None
        self._last_request_time = 0
        
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """生成文本内容
        
        Args:
            request: 标准化的LLM请求
            
        Returns:
            LLMResponse: 标准化的LLM响应
            
        Raises:
            LLMProviderError: 当生成失败时抛出
        """
        pass
    
    @abstractmethod
    async def generate_structured(self, request: LLMRequest, output_schema: Type) -> LLMResponse:
        """生成结构化输出
        
        Args:
            request: 标准化的LLM请求
            output_schema: 输出结构的Pydantic模型类
            
        Returns:
            LLMResponse: 包含结构化数据的响应
            
        Raises:
            LLMProviderError: 当生成失败时抛出
        """
        pass
    
    @abstractmethod
    def get_langchain_llm(self, model: str, **kwargs) -> BaseLanguageModel:
        """获取LangChain兼容的LLM实例
        
        Args:
            model: 模型名称
            **kwargs: 额外参数
            
        Returns:
            BaseLanguageModel: LangChain兼容的LLM实例
        """
        pass
    
    @abstractmethod
    def get_provider_type(self) -> LLMProviderType:
        """返回提供商类型
        
        Returns:
            LLMProviderType: 提供商的类型枚举
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
    async def get_available_models(self) -> List[LLMModel]:
        """获取可用模型列表
        
        Returns:
            List[LLMModel]: 可用模型列表
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
        if not self.validate_config():
            raise LLMProviderConfigError(
                f"Invalid configuration for {self.get_provider_type()}"
            )
    
    async def health_check(self) -> bool:
        """检查提供商服务是否可用
        
        Returns:
            bool: 服务是否可用
        """
        try:
            # 执行一个简单的测试请求
            test_request = LLMRequest(
                prompt="Hello",
                model=self.config.default_model,
                task_type=LLMTaskType.QUERY_GENERATION,
                temperature=0.1,
                max_tokens=10
            )
            await self.generate(test_request)
            return True
        except Exception as e:
            print(f"Health check failed for {self.get_provider_type()}: {e}")
            return False
    
    def get_default_model(self) -> str:
        """获取默认模型
        
        Returns:
            str: 默认模型名称
        """
        return self.config.default_model
    
    def get_available_model_names(self) -> List[str]:
        """获取可用模型名称列表
        
        Returns:
            List[str]: 模型名称列表
        """
        return self.config.models
    
    async def _rate_limit_check(self):
        """速率限制检查"""
        rate_limits = self.get_rate_limits()
        if "requests_per_second" in rate_limits:
            min_interval = 1.0 / rate_limits["requests_per_second"]
            current_time = time.time()
            time_since_last = current_time - self._last_request_time
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                await asyncio.sleep(sleep_time)
            
            self._last_request_time = time.time()
    
    def _prepare_request(self, request: LLMRequest) -> LLMRequest:
        """预处理请求"""
        # 确保模型在可用列表中
        if request.model not in self.config.models:
            print(f"Warning: Model {request.model} not in available models, using default {self.config.default_model}")
            request.model = self.config.default_model
        
        return request
    
    def _create_response(
        self, 
        content: str, 
        model: str, 
        usage: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        structured_data: Optional[Any] = None
    ) -> LLMResponse:
        """创建标准化响应"""
        return LLMResponse(
            content=content,
            model=model,
            provider=self.get_provider_type(),
            usage=usage or {},
            metadata=metadata or {},
            structured_data=structured_data
        )


class LLMProviderRegistry:
    """LLM提供商注册表
    
    管理所有已注册的LLM提供商类型。
    """
    
    _providers: Dict[LLMProviderType, Type[BaseLLMProvider]] = {}
    
    @classmethod
    def register(cls, provider_type: LLMProviderType, provider_class: Type[BaseLLMProvider]):
        """注册LLM提供商
        
        Args:
            provider_type: 提供商类型
            provider_class: 提供商类
        """
        if not issubclass(provider_class, BaseLLMProvider):
            raise ValueError(f"Provider class must inherit from BaseLLMProvider: {provider_class}")
        
        cls._providers[provider_type] = provider_class
    
    @classmethod
    def get_provider_class(cls, provider_type: LLMProviderType) -> Type[BaseLLMProvider]:
        """获取提供商类
        
        Args:
            provider_type: 提供商类型
            
        Returns:
            Type[BaseLLMProvider]: 提供商类
            
        Raises:
            LLMProviderConfigError: 当提供商未注册时
        """
        if provider_type not in cls._providers:
            available_types = list(cls._providers.keys())
            raise LLMProviderConfigError(
                f"Unknown provider type: {provider_type}. Available: {available_types}"
            )
        
        return cls._providers[provider_type]
    
    @classmethod
    def get_available_types(cls) -> List[LLMProviderType]:
        """获取所有可用的提供商类型
        
        Returns:
            List[LLMProviderType]: 可用提供商类型列表
        """
        return list(cls._providers.keys())