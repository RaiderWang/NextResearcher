"""LLM提供商标准化数据类型定义

定义了所有LLM提供商使用的标准化请求、响应和配置数据结构。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field
from enum import Enum


class LLMProviderType(str, Enum):
    """LLM提供商类型枚举"""
    GEMINI = "GOOGLE_GEMINI"
    AZURE_OPENAI = "AZURE_OPENAI"
    AWS_BEDROCK = "AWS_BEDROCK"
    OPENAI_COMPATIBLE = "OPENAI_COMPATIBLE"


class LLMTaskType(str, Enum):
    """LLM任务类型枚举"""
    QUERY_GENERATION = "query_generation"
    REFLECTION = "reflection"
    ANSWER_GENERATION = "answer_generation"


@dataclass
class LLMModel:
    """LLM模型信息"""
    id: str
    name: str
    provider: LLMProviderType
    max_tokens: Optional[int] = None
    supports_structured_output: bool = True
    cost_per_1k_tokens: Optional[float] = None
    description: Optional[str] = None


class LLMRequest(BaseModel):
    """标准化LLM请求"""
    prompt: str = Field(description="输入提示词")
    model: str = Field(description="使用的模型ID")
    task_type: LLMTaskType = Field(description="任务类型")
    temperature: float = Field(default=0.7, description="生成温度")
    max_tokens: Optional[int] = Field(default=None, description="最大生成token数")
    structured_output_schema: Optional[Dict[str, Any]] = Field(default=None, description="结构化输出模式")
    additional_params: Dict[str, Any] = Field(default_factory=dict, description="提供商特定参数")


class LLMResponse(BaseModel):
    """标准化LLM响应"""
    content: str = Field(description="生成的内容")
    model: str = Field(description="使用的模型ID")
    provider: LLMProviderType = Field(description="提供商类型")
    usage: Optional[Dict[str, Any]] = Field(default=None, description="使用统计")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="额外元数据")
    structured_data: Optional[Any] = Field(default=None, description="结构化输出数据")


class LLMProviderConfig(BaseModel):
    """LLM提供商配置基类"""
    provider_type: LLMProviderType = Field(description="提供商类型")
    api_key: str = Field(description="API密钥")
    models: List[str] = Field(description="可用模型列表")
    default_model: str = Field(description="默认模型")
    timeout: float = Field(default=30.0, description="请求超时时间")
    max_retries: int = Field(default=3, description="最大重试次数")
    additional_config: Dict[str, Any] = Field(default_factory=dict, description="提供商特定配置")


class GeminiProviderConfig(LLMProviderConfig):
    """Google Gemini提供商配置"""
    provider_type: LLMProviderType = Field(default=LLMProviderType.GEMINI)


class AzureOpenAIProviderConfig(LLMProviderConfig):
    """Azure OpenAI提供商配置"""
    provider_type: LLMProviderType = Field(default=LLMProviderType.AZURE_OPENAI)
    endpoint: str = Field(description="Azure OpenAI端点")
    api_version: str = Field(description="API版本")


class BedrockProviderConfig(LLMProviderConfig):
    """AWS Bedrock提供商配置"""
    provider_type: LLMProviderType = Field(default=LLMProviderType.AWS_BEDROCK)
    region: str = Field(description="AWS区域")
    access_key_id: str = Field(description="AWS访问密钥ID")
    secret_access_key: str = Field(description="AWS秘密访问密钥")


class OpenAICompatibleProviderConfig(LLMProviderConfig):
    """OpenAI兼容提供商配置"""
    provider_type: LLMProviderType = Field(default=LLMProviderType.OPENAI_COMPATIBLE)
    base_url: str = Field(description="API基础URL")


class LLMProviderError(Exception):
    """LLM提供商基础异常"""
    pass


class LLMProviderConfigError(LLMProviderError):
    """LLM提供商配置错误"""
    pass


class LLMProviderAPIError(LLMProviderError):
    """LLM提供商API错误"""
    pass


class LLMProviderTimeoutError(LLMProviderError):
    """LLM提供商超时错误"""
    pass


class LLMProviderRateLimitError(LLMProviderError):
    """LLM提供商速率限制错误"""
    pass