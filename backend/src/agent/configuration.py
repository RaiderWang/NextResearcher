"""Agent配置系统

支持多LLM提供商的统一配置管理。
"""

import os
from pydantic import BaseModel, Field
from typing import Any, Optional, Dict, List
from langchain_core.runnables import RunnableConfig

from .llm_types import LLMProviderType


class Configuration(BaseModel):
    """Agent配置类
    
    统一管理所有配置参数，包括LLM提供商、搜索提供商等。
    """

    # LLM提供商配置
    llm_provider: LLMProviderType = Field(
        default=LLMProviderType.GEMINI,
        metadata={
            "description": "LLM提供商类型: gemini, azure_openai, aws_bedrock, openai_compatible"
        },
    )

    query_generator_model: str = Field(
        default="gemini-2.0-flash",
        metadata={
            "description": "查询生成使用的模型"
        },
    )

    reflection_model: str = Field(
        default="gemini-2.5-flash",
        metadata={
            "description": "反思分析使用的模型"
        },
    )

    answer_model: str = Field(
        default="gemini-2.5-pro",
        metadata={
            "description": "答案生成使用的模型"
        },
    )

    # 搜索配置
    search_provider: str = Field(
        default="google",
        metadata={"description": "搜索提供商: google, tavily"}
    )

    search_results_limit: int = Field(
        default=10,
        metadata={"description": "每次搜索返回的结果数量限制"}
    )

    # 研究流程配置
    number_of_initial_queries: int = Field(
        default=3,
        metadata={"description": "初始搜索查询数量"},
    )

    max_research_loops: int = Field(
        default=2,
        metadata={"description": "最大研究循环次数"},
    )

    # LLM生成参数
    temperature: float = Field(
        default=0.7,
        metadata={"description": "LLM生成温度"}
    )

    max_tokens: Optional[int] = Field(
        default=None,
        metadata={"description": "最大生成token数"}
    )

    # 超时和重试配置
    request_timeout: float = Field(
        default=30.0,
        metadata={"description": "请求超时时间（秒）"}
    )

    max_retries: int = Field(
        default=3,
        metadata={"description": "最大重试次数"}
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """从RunnableConfig创建Configuration实例
        
        Args:
            config: LangGraph运行时配置
            
        Returns:
            Configuration: 配置实例
        """
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # 获取原始值，优先级：运行时配置 > 环境变量 > 默认值
        raw_values: Dict[str, Any] = {}
        for name in cls.model_fields.keys():
            # 1. 运行时配置（来自前端）
            runtime_value = configurable.get(name)
            # 2. 环境变量
            env_value = os.environ.get(name.upper())
            # 3. 使用运行时配置，如果没有则使用环境变量
            raw_values[name] = runtime_value if runtime_value is not None else env_value

        # 过滤掉None值和空字符串（将空字符串视为None）
        values = {k: v for k, v in raw_values.items() if v is not None and v != ""}

        # 特殊处理LLM提供商类型
        if "llm_provider" in values:
            try:
                values["llm_provider"] = LLMProviderType(values["llm_provider"])
            except ValueError:
                print(f"Warning: Invalid LLM provider '{values['llm_provider']}', using default")
                values.pop("llm_provider", None)

        return cls(**values)

    @classmethod
    def from_environment(cls) -> "Configuration":
        """从环境变量创建Configuration实例
        
        Returns:
            Configuration: 配置实例
        """
        return cls.from_runnable_config(None)

    def get_llm_config(self) -> Dict[str, Any]:
        """获取LLM相关配置
        
        Returns:
            Dict[str, Any]: LLM配置字典
        """
        return {
            "provider": self.llm_provider,
            "query_generator_model": self.query_generator_model,
            "reflection_model": self.reflection_model,
            "answer_model": self.answer_model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "request_timeout": self.request_timeout,
            "max_retries": self.max_retries
        }

    def get_search_config(self) -> Dict[str, Any]:
        """获取搜索相关配置
        
        Returns:
            Dict[str, Any]: 搜索配置字典
        """
        return {
            "provider": self.search_provider,
            "results_limit": self.search_results_limit
        }

    def get_research_config(self) -> Dict[str, Any]:
        """获取研究流程配置
        
        Returns:
            Dict[str, Any]: 研究配置字典
        """
        return {
            "initial_queries": self.number_of_initial_queries,
            "max_loops": self.max_research_loops
        }

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return self.model_dump()

    def validate_llm_config(self) -> tuple[bool, List[str]]:
        """验证LLM配置
        
        Returns:
            tuple[bool, List[str]]: (是否有效, 错误信息列表)
        """
        errors = []
        
        # 验证提供商类型
        if not isinstance(self.llm_provider, LLMProviderType):
            errors.append(f"Invalid LLM provider type: {self.llm_provider}")
        
        # 验证模型名称
        if not self.query_generator_model:
            errors.append("Query generator model is required")
        if not self.reflection_model:
            errors.append("Reflection model is required")
        if not self.answer_model:
            errors.append("Answer model is required")
        
        # 验证数值范围
        if not (0.0 <= self.temperature <= 2.0):
            errors.append("Temperature must be between 0.0 and 2.0")
        
        if self.max_tokens is not None and self.max_tokens <= 0:
            errors.append("Max tokens must be positive")
        
        if self.request_timeout <= 0:
            errors.append("Request timeout must be positive")
        
        if self.max_retries < 0:
            errors.append("Max retries must be non-negative")
        
        return len(errors) == 0, errors


class ModelConfiguration:
    """模型配置管理器
    
    管理不同提供商的可用模型列表。
    """
    
    @staticmethod
    def get_available_models_by_provider() -> Dict[LLMProviderType, List[str]]:
        """获取各提供商的可用模型列表
        
        Returns:
            Dict[LLMProviderType, List[str]]: 按提供商分组的模型列表
        """
        models = {}
        
        # Gemini模型
        gemini_models_str = os.getenv("GEMINI_MODELS", "gemini-2.0-flash,gemini-2.5-flash,gemini-2.5-pro")
        models[LLMProviderType.GEMINI] = [model.strip() for model in gemini_models_str.split(",")]
        
        # Azure OpenAI模型
        azure_models_str = os.getenv("AZURE_OPENAI_MODELS", "gpt-4,gpt-4-turbo,gpt-35-turbo")
        models[LLMProviderType.AZURE_OPENAI] = [model.strip() for model in azure_models_str.split(",")]
        
        # AWS Bedrock模型
        bedrock_models_str = os.getenv("BEDROCK_MODELS", "anthropic.claude-3-sonnet-20240229-v1:0,anthropic.claude-3-haiku-20240307-v1:0")
        models[LLMProviderType.AWS_BEDROCK] = [model.strip() for model in bedrock_models_str.split(",")]
        
        # OpenAI兼容模型
        openai_models_str = os.getenv("OPENAI_COMPATIBLE_MODELS", "gpt-3.5-turbo,gpt-4")
        models[LLMProviderType.OPENAI_COMPATIBLE] = [model.strip() for model in openai_models_str.split(",")]
        
        return models
    
    @staticmethod
    def get_default_models_by_provider() -> Dict[LLMProviderType, str]:
        """获取各提供商的默认模型
        
        Returns:
            Dict[LLMProviderType, str]: 各提供商的默认模型
        """
        models = ModelConfiguration.get_available_models_by_provider()
        defaults = {}
        
        for provider, model_list in models.items():
            if model_list:
                # 使用环境变量指定的默认模型，或列表中的第一个
                if provider == LLMProviderType.GEMINI:
                    defaults[provider] = os.getenv("GEMINI_DEFAULT_MODEL", model_list[0])
                elif provider == LLMProviderType.AZURE_OPENAI:
                    defaults[provider] = os.getenv("AZURE_OPENAI_DEFAULT_MODEL", model_list[0])
                elif provider == LLMProviderType.AWS_BEDROCK:
                    defaults[provider] = os.getenv("BEDROCK_DEFAULT_MODEL", model_list[0])
                elif provider == LLMProviderType.OPENAI_COMPATIBLE:
                    defaults[provider] = os.getenv("OPENAI_COMPATIBLE_DEFAULT_MODEL", model_list[0])
        
        return defaults
    
    @staticmethod
    def validate_model_for_provider(provider: LLMProviderType, model: str) -> bool:
        """验证模型是否适用于指定提供商
        
        Args:
            provider: 提供商类型
            model: 模型名称
            
        Returns:
            bool: 是否有效
        """
        available_models = ModelConfiguration.get_available_models_by_provider()
        return model in available_models.get(provider, [])
