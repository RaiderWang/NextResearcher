"""LLM服务层

提供统一的LLM调用接口，简化graph.py中的LLM使用。
"""

import os
from typing import Dict, Any, List, Optional, Type, Union
from langchain_core.language_models import BaseLanguageModel
from langchain_core.runnables import RunnableConfig

# 确保提供商注册
from . import provider_registry

from .llm_factory import LLMProviderFactory
from .llm_providers import BaseLLMProvider
from .llm_types import (
    LLMRequest,
    LLMResponse,
    LLMProviderType,
    LLMTaskType,
    LLMModel,
    LLMProviderError
)


class LLMService:
    """LLM服务类
    
    提供统一的LLM调用接口，管理不同提供商的LLM实例。
    """
    
    def __init__(self):
        """初始化LLM服务"""
        self._factory = LLMProviderFactory()
        self._default_provider: Optional[BaseLLMProvider] = None
        self._providers_cache: Dict[str, BaseLLMProvider] = {}
    
    def get_default_provider(self) -> BaseLLMProvider:
        """获取默认LLM提供商
        
        Returns:
            BaseLLMProvider: 默认提供商实例
        """
        if self._default_provider is None:
            self._default_provider = self._factory.create_default_provider()
        return self._default_provider
    
    def get_provider(self, provider_type: LLMProviderType) -> BaseLLMProvider:
        """获取指定类型的LLM提供商
        
        Args:
            provider_type: 提供商类型
            
        Returns:
            BaseLLMProvider: 提供商实例
        """
        cache_key = str(provider_type)
        if cache_key not in self._providers_cache:
            self._providers_cache[cache_key] = self._factory.create_provider(provider_type)
        return self._providers_cache[cache_key]
    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider_type: Optional[LLMProviderType] = None,
        task_type: LLMTaskType = LLMTaskType.QUERY_GENERATION,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """生成文本内容
        
        Args:
            prompt: 输入提示词
            model: 模型名称，如果为None则使用默认模型
            provider_type: 提供商类型，如果为None则使用默认提供商
            task_type: 任务类型
            temperature: 生成温度
            max_tokens: 最大token数
            **kwargs: 额外参数
            
        Returns:
            LLMResponse: 生成响应
        """
        # 获取提供商
        if provider_type:
            provider = self.get_provider(provider_type)
        else:
            provider = self.get_default_provider()
        
        # 使用提供商的默认模型（如果未指定）
        if not model:
            model = provider.get_default_model()
        
        # 创建请求
        request = LLMRequest(
            prompt=prompt,
            model=model,
            task_type=task_type,
            temperature=temperature,
            max_tokens=max_tokens,
            additional_params=kwargs
        )
        
        return await provider.generate(request)
    
    async def generate_structured(
        self,
        prompt: str,
        output_schema: Type,
        model: Optional[str] = None,
        provider_type: Optional[LLMProviderType] = None,
        task_type: LLMTaskType = LLMTaskType.QUERY_GENERATION,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """生成结构化输出
        
        Args:
            prompt: 输入提示词
            output_schema: 输出结构的Pydantic模型类
            model: 模型名称，如果为None则使用默认模型
            provider_type: 提供商类型，如果为None则使用默认提供商
            task_type: 任务类型
            temperature: 生成温度
            max_tokens: 最大token数
            **kwargs: 额外参数
            
        Returns:
            LLMResponse: 包含结构化数据的响应
        """
        # 获取提供商
        if provider_type:
            provider = self.get_provider(provider_type)
        else:
            provider = self.get_default_provider()
        
        # 使用提供商的默认模型（如果未指定）
        if not model:
            model = provider.get_default_model()
        
        # 创建请求
        request = LLMRequest(
            prompt=prompt,
            model=model,
            task_type=task_type,
            temperature=temperature,
            max_tokens=max_tokens,
            structured_output_schema=output_schema.model_json_schema() if hasattr(output_schema, 'model_json_schema') else None,
            additional_params=kwargs
        )
        
        return await provider.generate_structured(request, output_schema)
    
    def get_langchain_llm(
        self,
        model: Optional[str] = None,
        provider_type: Optional[LLMProviderType] = None,
        **kwargs
    ) -> BaseLanguageModel:
        """获取LangChain兼容的LLM实例
        
        Args:
            model: 模型名称，如果为None则使用默认模型
            provider_type: 提供商类型，如果为None则使用默认提供商
            **kwargs: 额外参数
            
        Returns:
            BaseLanguageModel: LangChain兼容的LLM实例
        """
        # 获取提供商
        if provider_type:
            provider = self.get_provider(provider_type)
        else:
            provider = self.get_default_provider()
        
        # 使用提供商的默认模型（如果未指定）
        if not model:
            model = provider.get_default_model()
        
        return provider.get_langchain_llm(model, **kwargs)
    
    async def get_all_available_models(self) -> Dict[LLMProviderType, List[LLMModel]]:
        """获取所有提供商的可用模型
        
        Returns:
            Dict[LLMProviderType, List[LLMModel]]: 按提供商分组的模型列表
        """
        return await self._factory.get_all_available_models()
    
    def get_available_providers(self) -> List[LLMProviderType]:
        """获取所有可用的提供商类型
        
        Returns:
            List[LLMProviderType]: 可用提供商类型列表
        """
        return self._factory.get_available_providers()
    
    async def health_check_all(self) -> Dict[LLMProviderType, bool]:
        """检查所有提供商的健康状态
        
        Returns:
            Dict[LLMProviderType, bool]: 各提供商的健康状态
        """
        return await self._factory.health_check_all()
    
    def get_provider_info(self, provider_type: LLMProviderType) -> Dict[str, Any]:
        """获取提供商信息
        
        Args:
            provider_type: 提供商类型
            
        Returns:
            Dict[str, Any]: 提供商信息
        """
        try:
            provider = self.get_provider(provider_type)
            return {
                "type": provider.get_provider_type(),
                "default_model": provider.get_default_model(),
                "available_models": provider.get_available_model_names(),
                "rate_limits": provider.get_rate_limits(),
                "config_valid": provider.validate_config()
            }
        except Exception as e:
            return {
                "type": provider_type,
                "error": str(e),
                "config_valid": False
            }


class ConfigurableLLMService(LLMService):
    """可配置的LLM服务
    
    支持从RunnableConfig中读取配置参数。
    """
    
    def __init__(self, config: Optional[RunnableConfig] = None):
        """初始化可配置LLM服务
        
        Args:
            config: 运行时配置
        """
        super().__init__()
        self.config = config or {}
        self._extract_config_params()
    
    def _extract_config_params(self):
        """从配置中提取参数"""
        configurable = self.config.get("configurable", {})
        
        # 添加完整的配置调试信息
        print(f"🔧 ConfigurableLLMService: 配置解析")
        print(f"🔧 configurable keys: {list(configurable.keys()) if configurable else 'None'}")
        print(f"🔧 reasoning_model in configurable: {configurable.get('reasoning_model') if configurable else 'N/A'}")
        print(f"🔧 config top-level keys: {list(self.config.keys()) if self.config else 'None'}")
        print(f"🔧 reasoning_model in config: {self.config.get('reasoning_model') if self.config else 'N/A'}")
        
        # 使用与Configuration相同的参数读取逻辑
        def get_param_value(param_name: str):
            """获取参数值，优先级：configurable > 顶层config > 环境变量"""
            # 1. 从configurable中获取
            value = configurable.get(param_name)
            if value is not None:
                return value
            
            # 2. 从顶层config中获取
            value = self.config.get(param_name)
            if value is not None:
                return value
            
            # 3. 从环境变量获取
            import os
            return os.environ.get(param_name.upper())
        
        # 提取LLM相关配置
        self.query_generator_model = get_param_value("query_generator_model")
        self.reflection_model = get_param_value("reflection_model")
        self.answer_model = get_param_value("answer_model")
        self.llm_provider = get_param_value("llm_provider")
        
        # 如果用户设置了reasoning_model，用它来覆盖所有任务模型
        reasoning_model = get_param_value("reasoning_model")
        if reasoning_model:
            print(f"🔧 用户选择了reasoning_model: {reasoning_model}，将用于所有任务")
            self.query_generator_model = reasoning_model
            self.reflection_model = reasoning_model  
            self.answer_model = reasoning_model
        
        print(f"🔧 最终参数:")
        print(f"🔧 - query_generator_model: {self.query_generator_model}")
        print(f"🔧 - reflection_model: {self.reflection_model}")
        print(f"🔧 - answer_model: {self.answer_model}")
        print(f"🔧 - llm_provider: {self.llm_provider}")
        print(f"🔧 - reasoning_model (用户选择): {reasoning_model}")
        
        # 如果指定了提供商，尝试获取对应的提供商实例
        if self.llm_provider:
            try:
                provider_type = LLMProviderType(self.llm_provider)
                self._default_provider = self.get_provider(provider_type)
                print(f"🔧 成功设置提供商: {provider_type}")
            except (ValueError, LLMProviderError) as e:
                print(f"Warning: Failed to set provider {self.llm_provider}: {e}")
        else:
            print(f"🔧 未指定llm_provider，使用系统默认提供商")
    
    def get_model_for_task(self, task_type: LLMTaskType) -> Optional[str]:
        """根据任务类型获取对应的模型
        
        Args:
            task_type: 任务类型
            
        Returns:
            Optional[str]: 模型名称
        """
        if task_type == LLMTaskType.QUERY_GENERATION:
            return self.query_generator_model
        elif task_type == LLMTaskType.REFLECTION:
            return self.reflection_model
        elif task_type == LLMTaskType.ANSWER_GENERATION:
            return self.answer_model
        else:
            return None
    
    async def generate_for_task(
        self,
        prompt: str,
        task_type: LLMTaskType,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """根据任务类型生成内容
        
        Args:
            prompt: 输入提示词
            task_type: 任务类型
            temperature: 生成温度
            max_tokens: 最大token数
            **kwargs: 额外参数
            
        Returns:
            LLMResponse: 生成响应
        """
        model = self.get_model_for_task(task_type)
        return await self.generate(
            prompt=prompt,
            model=model,
            task_type=task_type,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    async def generate_structured_for_task(
        self,
        prompt: str,
        output_schema: Type,
        task_type: LLMTaskType,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """根据任务类型生成结构化内容
        
        Args:
            prompt: 输入提示词
            output_schema: 输出结构的Pydantic模型类
            task_type: 任务类型
            temperature: 生成温度
            max_tokens: 最大token数
            **kwargs: 额外参数
            
        Returns:
            LLMResponse: 包含结构化数据的响应
        """
        model = self.get_model_for_task(task_type)
        return await self.generate_structured(
            prompt=prompt,
            output_schema=output_schema,
            model=model,
            task_type=task_type,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def get_default_provider(self) -> BaseLLMProvider:
        """获取默认LLM提供商实例
        
        重写父类方法，优先使用配置指定的提供商
        
        Returns:
            BaseLLMProvider: 默认提供商实例
        """
        # 如果配置中指定了提供商，使用配置的提供商
        if hasattr(self, '_default_provider') and self._default_provider is not None:
            print(f"🔧 get_default_provider: 使用配置的提供商: {self._default_provider.get_provider_type()}")
            return self._default_provider
        
        # 否则使用父类的默认提供商
        print(f"🔧 get_default_provider: 使用系统默认提供商")
        return super().get_default_provider()


# 全局LLM服务实例
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """获取全局LLM服务实例
    
    Returns:
        LLMService: LLM服务实例
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


def create_configurable_llm_service(config: RunnableConfig) -> ConfigurableLLMService:
    """创建可配置的LLM服务实例
    
    Args:
        config: 运行时配置
        
    Returns:
        ConfigurableLLMService: 可配置LLM服务实例
    """
    return ConfigurableLLMService(config)