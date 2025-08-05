"""LLM提供商工厂类

负责创建和管理不同的LLM提供商实例。
"""

import os
from typing import Dict, Any, List, Optional, Type

# 导入provider_registry以确保所有提供商都被注册
from . import provider_registry

from .llm_types import (
    LLMProviderType,
    LLMProviderConfig,
    GeminiProviderConfig,
    AzureOpenAIProviderConfig,
    BedrockProviderConfig,
    OpenAICompatibleProviderConfig,
    LLMProviderConfigError,
    LLMModel
)
from .llm_providers import BaseLLMProvider, LLMProviderRegistry


class LLMProviderFactory:
    """LLM提供商工厂类
    
    负责创建和管理不同的LLM提供商实例。
    """
    
    _instance: Optional['LLMProviderFactory'] = None
    _providers_cache: Dict[str, BaseLLMProvider] = {}
    
    def __new__(cls) -> 'LLMProviderFactory':
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    @classmethod
    def create_provider(
        cls, 
        provider_type: LLMProviderType, 
        config: Optional[LLMProviderConfig] = None
    ) -> BaseLLMProvider:
        """根据配置创建LLM提供商实例
        
        Args:
            provider_type: 提供商类型
            config: 提供商配置，如果为None则从环境变量获取
            
        Returns:
            BaseLLMProvider: LLM提供商实例
            
        Raises:
            LLMProviderConfigError: 当提供商不存在或配置无效时
        """
        # 使用缓存避免重复创建
        cache_key = f"{provider_type}_{hash(str(config))}"
        if cache_key in cls._providers_cache:
            return cls._providers_cache[cache_key]
        
        # 如果没有提供配置，从环境变量获取
        if config is None:
            config = cls._get_default_config(provider_type)
        
        # 获取提供商类并创建实例
        provider_class = LLMProviderRegistry.get_provider_class(provider_type)
        provider = provider_class(config)
        
        # 缓存实例
        cls._providers_cache[cache_key] = provider
        
        return provider
    
    @classmethod
    def get_available_providers(cls) -> List[LLMProviderType]:
        """获取所有可用的LLM提供商列表
        
        Returns:
            List[LLMProviderType]: 可用提供商类型列表
        """
        return LLMProviderRegistry.get_available_types()
    
    @classmethod
    def get_default_provider_type(cls) -> LLMProviderType:
        """获取默认提供商类型
        
        Returns:
            LLMProviderType: 默认提供商类型
        """
        provider_name = os.getenv("LLM_PROVIDER", "GOOGLE_GEMINI").upper()
        try:
            return LLMProviderType(provider_name)
        except ValueError:
            print(f"Warning: Unknown provider '{provider_name}', using Gemini as default")
            return LLMProviderType.GEMINI
    
    @classmethod
    def create_default_provider(cls) -> BaseLLMProvider:
        """创建默认提供商实例
        
        Returns:
            BaseLLMProvider: 默认提供商实例
        """
        provider_type = cls.get_default_provider_type()
        return cls.create_provider(provider_type)
    
    @classmethod
    def _get_default_config(cls, provider_type: LLMProviderType) -> LLMProviderConfig:
        """从环境变量获取默认配置
        
        Args:
            provider_type: 提供商类型
            
        Returns:
            LLMProviderConfig: 默认配置
            
        Raises:
            LLMProviderConfigError: 当必需的环境变量缺失时
        """
        if provider_type == LLMProviderType.GEMINI:
            return cls._get_gemini_config()
        elif provider_type == LLMProviderType.AZURE_OPENAI:
            return cls._get_azure_openai_config()
        elif provider_type == LLMProviderType.AWS_BEDROCK:
            return cls._get_bedrock_config()
        elif provider_type == LLMProviderType.OPENAI_COMPATIBLE:
            return cls._get_openai_compatible_config()
        else:
            raise LLMProviderConfigError(f"Unsupported provider type: {provider_type}")
    
    @classmethod
    def _get_gemini_config(cls) -> GeminiProviderConfig:
        """获取Gemini配置"""
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMProviderConfigError("GEMINI_API_KEY environment variable is required")
        
        models_str = os.getenv("GEMINI_MODELS", "gemini-2.0-flash,gemini-2.5-flash,gemini-2.5-pro")
        models = [model.strip() for model in models_str.split(",")]
        default_model = os.getenv("GEMINI_DEFAULT_MODEL", models[0])
        
        return GeminiProviderConfig(
            api_key=api_key,
            models=models,
            default_model=default_model,
            timeout=float(os.getenv("GEMINI_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("GEMINI_MAX_RETRIES", "3"))
        )
    
    @classmethod
    def _get_azure_openai_config(cls) -> AzureOpenAIProviderConfig:
        """获取Azure OpenAI配置"""
        api_key = os.getenv("AZURE_OPENAI_API_KEY")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        
        if not all([api_key, endpoint, api_version]):
            raise LLMProviderConfigError(
                "AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, and AZURE_OPENAI_API_VERSION "
                "environment variables are required"
            )
        
        models_str = os.getenv("AZURE_OPENAI_MODELS", "gpt-4,gpt-4-turbo,gpt-35-turbo")
        models = [model.strip() for model in models_str.split(",")]
        default_model = os.getenv("AZURE_OPENAI_DEFAULT_MODEL", models[0])
        
        return AzureOpenAIProviderConfig(
            api_key=api_key,
            endpoint=endpoint,
            api_version=api_version,
            models=models,
            default_model=default_model,
            timeout=float(os.getenv("AZURE_OPENAI_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("AZURE_OPENAI_MAX_RETRIES", "3"))
        )
    
    @classmethod
    def _get_bedrock_config(cls) -> BedrockProviderConfig:
        """获取AWS Bedrock配置"""
        access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
        secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
        region = os.getenv("AWS_REGION")
        
        if not all([access_key_id, secret_access_key, region]):
            raise LLMProviderConfigError(
                "AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, and AWS_REGION "
                "environment variables are required"
            )
        
        models_str = os.getenv("BEDROCK_MODELS", "anthropic.claude-3-sonnet-20240229-v1:0,anthropic.claude-3-haiku-20240307-v1:0")
        models = [model.strip() for model in models_str.split(",")]
        default_model = os.getenv("BEDROCK_DEFAULT_MODEL", models[0])
        
        return BedrockProviderConfig(
            api_key="",  # Bedrock uses AWS credentials
            region=region,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            models=models,
            default_model=default_model,
            timeout=float(os.getenv("BEDROCK_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("BEDROCK_MAX_RETRIES", "3"))
        )
    
    @classmethod
    def _get_openai_compatible_config(cls) -> OpenAICompatibleProviderConfig:
        """获取OpenAI兼容配置"""
        api_key = os.getenv("OPENAI_COMPATIBLE_API_KEY")
        base_url = os.getenv("OPENAI_COMPATIBLE_BASE_URL")
        
        # 对于OpenAI兼容提供商，至少需要base_url
        # api_key对于本地模型可能是可选的
        if not base_url:
            raise LLMProviderConfigError(
                "OPENAI_BASE_URL environment variable is required for OpenAI Compatible provider"
            )
        
        models_str = os.getenv("OPENAI_COMPATIBLE_MODELS", "gpt-3.5-turbo,gpt-4")
        models = [model.strip() for model in models_str.split(",")]
        default_model = os.getenv("OPENAI_COMPATIBLE_DEFAULT_MODEL", models[0])
        
        return OpenAICompatibleProviderConfig(
            api_key=api_key or "",  # 允许空的API key用于本地模型
            base_url=base_url,
            models=models,
            default_model=default_model,
            timeout=float(os.getenv("OPENAI_COMPATIBLE_TIMEOUT", "30.0")),
            max_retries=int(os.getenv("OPENAI_COMPATIBLE_MAX_RETRIES", "3"))
        )
    
    @classmethod
    async def get_all_available_models(cls) -> Dict[LLMProviderType, List[LLMModel]]:
        """获取所有已配置提供商的可用模型
        
        只返回配置完整且可用的提供商模型列表
        
        Returns:
            Dict[LLMProviderType, List[LLMModel]]: 按提供商分组的模型列表
        """
        all_models = {}
        
        for provider_type in cls.get_available_providers():
            # 检查提供商是否已正确配置
            if not cls.is_provider_available(provider_type):
                print(f"Skipping {provider_type}: not properly configured")
                continue
                
            try:
                provider = cls.create_provider(provider_type)
                models = await provider.get_available_models()
                if models:  # 只包含有模型的提供商
                    all_models[provider_type] = models
            except Exception as e:
                print(f"Failed to get models for {provider_type}: {e}")
                # 不包含失败的提供商，而不是返回空列表
        
        return all_models
    
    @classmethod
    def validate_provider_config(
        cls, 
        provider_type: LLMProviderType, 
        config: Optional[LLMProviderConfig] = None
    ) -> tuple[bool, str]:
        """验证提供商配置
        
        Args:
            provider_type: 提供商类型
            config: 配置，如果为None则验证环境变量配置
            
        Returns:
            tuple[bool, str]: (是否有效, 错误信息)
        """
        try:
            provider = cls.create_provider(provider_type, config)
            return True, "Configuration is valid"
        except Exception as e:
            return False, str(e)
    
    @classmethod
    async def health_check_all(cls) -> Dict[LLMProviderType, bool]:
        """检查所有提供商的健康状态
        
        Returns:
            Dict[LLMProviderType, bool]: 各提供商的健康状态
        """
        health_status = {}
        
        for provider_type in cls.get_available_providers():
            try:
                provider = cls.create_provider(provider_type)
                is_healthy = await provider.health_check()
                health_status[provider_type] = is_healthy
            except Exception as e:
                print(f"Health check failed for {provider_type}: {e}")
                health_status[provider_type] = False
        
        return health_status
    
    @classmethod
    def is_provider_available(cls, provider_type: LLMProviderType) -> bool:
        """检查提供商是否可用（配置是否完整）
        
        Args:
            provider_type: 提供商类型
            
        Returns:
            bool: 提供商是否可用
        """
        try:
            # 尝试获取配置，如果配置不完整会抛出异常
            config = cls._get_default_config(provider_type)
            return True
        except LLMProviderConfigError as e:
            print(f"Provider {provider_type} not available: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error checking provider {provider_type}: {e}")
            return False