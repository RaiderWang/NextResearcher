import os
from typing import Dict, Any, List, Type, Optional
from agent.search_providers import BaseSearchProvider, SearchProviderConfigError
from agent.providers.google_search_provider import GoogleSearchProvider
from agent.providers.tavily_search_provider import TavilySearchProvider


class SearchProviderFactory:
    """搜索提供商工厂类
    
    负责创建和管理不同的搜索提供商实例。
    """
    
    # 注册的搜索提供商
    _providers: Dict[str, Type[BaseSearchProvider]] = {
        "google": GoogleSearchProvider,
        "tavily": TavilySearchProvider,
    }
    
    @classmethod
    def create_provider(
        cls, 
        provider_name: str, 
        config: Optional[Dict[str, Any]] = None
    ) -> BaseSearchProvider:
        """根据配置创建搜索提供商实例
        
        Args:
            provider_name: 提供商名称
            config: 提供商配置，如果为None则从环境变量获取
            
        Returns:
            BaseSearchProvider: 搜索提供商实例
            
        Raises:
            SearchProviderConfigError: 当提供商不存在或配置无效时
        """
        if provider_name not in cls._providers:
            available_providers = list(cls._providers.keys())
            raise SearchProviderConfigError(
                f"未知的搜索提供商: {provider_name}. "
                f"可用的提供商: {available_providers}"
            )
        
        # 如果没有提供配置，从环境变量获取
        if config is None:
            config = cls._get_default_config(provider_name)
        
        provider_class = cls._providers[provider_name]
        return provider_class(config)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """获取所有可用的搜索提供商列表
        
        Returns:
            List[str]: 可用提供商名称列表
        """
        return list(cls._providers.keys())
    
    @classmethod
    def register_provider(
        cls, 
        name: str, 
        provider_class: Type[BaseSearchProvider]
    ):
        """注册新的搜索提供商
        
        Args:
            name: 提供商名称
            provider_class: 提供商类
        """
        if not issubclass(provider_class, BaseSearchProvider):
            raise ValueError(
                f"提供商类必须继承自BaseSearchProvider: {provider_class}"
            )
        
        cls._providers[name] = provider_class
    
    @classmethod
    def _get_default_config(cls, provider_name: str) -> Dict[str, Any]:
        """从环境变量获取默认配置
        
        Args:
            provider_name: 提供商名称
            
        Returns:
            Dict[str, Any]: 默认配置字典
        """
        config = {}
        
        if provider_name == "google":
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise SearchProviderConfigError(
                    "未找到GEMINI_API_KEY环境变量，Google搜索提供商需要此密钥"
                )
            config = {
                "api_key": api_key,
                "model": os.getenv("GOOGLE_SEARCH_MODEL", "gemini-2.0-flash")
            }
        elif provider_name == "tavily":
            api_key = os.getenv("TAVILY_API_KEY")
            if not api_key:
                raise SearchProviderConfigError(
                    "未找到TAVILY_API_KEY环境变量，Tavily搜索提供商需要此密钥"
                )
            config = {
                "api_key": api_key,
                "base_url": os.getenv("TAVILY_BASE_URL", "https://api.tavily.com"),
                "timeout": float(os.getenv("TAVILY_TIMEOUT", "30.0"))
            }
        
        return config
    
    @classmethod
    def get_provider_info(cls, provider_name: str) -> Dict[str, Any]:
        """获取提供商信息
        
        Args:
            provider_name: 提供商名称
            
        Returns:
            Dict[str, Any]: 提供商信息
        """
        if provider_name not in cls._providers:
            raise SearchProviderConfigError(f"未知的搜索提供商: {provider_name}")
        
        provider_class = cls._providers[provider_name]
        
        # 创建临时实例以获取信息（使用空配置）
        try:
            temp_config = cls._get_default_config(provider_name)
            temp_instance = provider_class(temp_config)
            
            return {
                "name": provider_name,
                "class_name": provider_class.__name__,
                "required_config_keys": temp_instance.get_required_config_keys(),
                "rate_limits": temp_instance.get_rate_limits(),
                "supported_features": getattr(temp_instance, 'get_supported_features', lambda: [])(),
                "description": provider_class.__doc__ or "无描述",
            }
        except Exception as e:
            return {
                "name": provider_name,
                "class_name": provider_class.__name__,
                "error": f"无法获取提供商信息: {e}",
                "description": provider_class.__doc__ or "无描述",
            }
    
    @classmethod
    def validate_provider_config(
        cls, 
        provider_name: str, 
        config: Dict[str, Any]
    ) -> tuple[bool, str]:
        """验证提供商配置
        
        Args:
            provider_name: 提供商名称
            config: 配置字典
            
        Returns:
            tuple[bool, str]: (是否有效, 错误信息)
        """
        try:
            provider = cls.create_provider(provider_name, config)
            return True, "配置有效"
        except Exception as e:
            return False, str(e) 