"""Azure OpenAI LLM提供商实现

提供Azure OpenAI服务的标准化LLM接口。
"""

import os
from typing import Dict, Any, List, Optional, Type
from langchain_openai import AzureChatOpenAI
from langchain_core.language_models import BaseLanguageModel
import openai

from ..llm_providers import BaseLLMProvider, LLMProviderRegistry
from ..llm_types import (
    LLMRequest,
    LLMResponse,
    LLMProviderType,
    LLMModel,
    AzureOpenAIProviderConfig,
    LLMProviderError,
    LLMProviderAPIError,
    LLMProviderTimeoutError
)


class AzureOpenAIProvider(BaseLLMProvider):
    """Azure OpenAI LLM提供商
    
    封装Azure OpenAI API的调用，提供标准化接口。
    """
    
    def __init__(self, config: AzureOpenAIProviderConfig):
        """初始化Azure OpenAI提供商
        
        Args:
            config: Azure OpenAI配置
        """
        super().__init__(config)
        self.config: AzureOpenAIProviderConfig = config
        
        # 初始化Azure OpenAI客户端
        self._client = openai.AzureOpenAI(
            api_key=config.api_key,
            azure_endpoint=config.endpoint,
            api_version=config.api_version
        )
        
        # 预定义的模型信息
        self._model_info = {
            "gpt-4": LLMModel(
                id="gpt-4",
                name="GPT-4",
                provider=LLMProviderType.AZURE_OPENAI,
                max_tokens=8192,
                supports_structured_output=True,
                description="Advanced reasoning and complex task model"
            ),
            "gpt-4-turbo": LLMModel(
                id="gpt-4-turbo",
                name="GPT-4 Turbo",
                provider=LLMProviderType.AZURE_OPENAI,
                max_tokens=128000,
                supports_structured_output=True,
                description="Enhanced GPT-4 with larger context window"
            ),
            "gpt-35-turbo": LLMModel(
                id="gpt-35-turbo",
                name="GPT-3.5 Turbo",
                provider=LLMProviderType.AZURE_OPENAI,
                max_tokens=4096,
                supports_structured_output=True,
                description="Fast and efficient model for general tasks"
            ),
            "gpt-4o": LLMModel(
                id="gpt-4o",
                name="GPT-4o",
                provider=LLMProviderType.AZURE_OPENAI,
                max_tokens=128000,
                supports_structured_output=True,
                description="Optimized GPT-4 model with multimodal capabilities"
            )
        }
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """生成文本内容
        
        Args:
            request: 标准化的LLM请求
            
        Returns:
            LLMResponse: 标准化的LLM响应
        """
        await self._rate_limit_check()
        request = self._prepare_request(request)
        
        try:
            # 创建LangChain LLM实例
            llm = self.get_langchain_llm(
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # 调用LLM
            result = llm.invoke(request.prompt)
            
            # 提取内容
            content = result.content if hasattr(result, 'content') else str(result)
            
            # 提取使用统计
            usage = {}
            if hasattr(result, 'response_metadata') and result.response_metadata:
                token_usage = result.response_metadata.get('token_usage', {})
                usage = {
                    "prompt_tokens": token_usage.get('prompt_tokens', 0),
                    "completion_tokens": token_usage.get('completion_tokens', 0),
                    "total_tokens": token_usage.get('total_tokens', 0)
                }
            
            return self._create_response(
                content=content,
                model=request.model,
                usage=usage,
                metadata={
                    "task_type": request.task_type,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "endpoint": self.config.endpoint
                }
            )
            
        except Exception as e:
            raise LLMProviderAPIError(f"Azure OpenAI API error: {str(e)}")
    
    async def generate_structured(self, request: LLMRequest, output_schema: Type) -> LLMResponse:
        """生成结构化输出
        
        Args:
            request: 标准化的LLM请求
            output_schema: 输出结构的Pydantic模型类
            
        Returns:
            LLMResponse: 包含结构化数据的响应
        """
        await self._rate_limit_check()
        request = self._prepare_request(request)
        
        try:
            # 创建支持结构化输出的LLM实例
            llm = self.get_langchain_llm(
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # 使用结构化输出
            structured_llm = llm.with_structured_output(output_schema)
            result = structured_llm.invoke(request.prompt)
            
            # 生成文本表示
            content = str(result) if result else ""
            
            return self._create_response(
                content=content,
                model=request.model,
                structured_data=result,
                metadata={
                    "task_type": request.task_type,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "structured_output": True,
                    "schema": output_schema.__name__,
                    "endpoint": self.config.endpoint
                }
            )
            
        except Exception as e:
            raise LLMProviderAPIError(f"Azure OpenAI structured output error: {str(e)}")
    
    def get_langchain_llm(self, model: str, **kwargs) -> BaseLanguageModel:
        """获取LangChain兼容的LLM实例
        
        Args:
            model: 模型名称
            **kwargs: 额外参数
            
        Returns:
            BaseLanguageModel: LangChain兼容的LLM实例
        """
        return AzureChatOpenAI(
            azure_deployment=model,  # Azure中使用deployment名称
            api_key=self.config.api_key,
            azure_endpoint=self.config.endpoint,
            api_version=self.config.api_version,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
            **kwargs
        )
    
    def get_provider_type(self) -> LLMProviderType:
        """返回提供商类型"""
        return LLMProviderType.AZURE_OPENAI
    
    def validate_config(self) -> bool:
        """验证提供商配置是否有效"""
        try:
            # 检查必需的配置项
            required_fields = ['api_key', 'endpoint', 'api_version']
            for field in required_fields:
                if not getattr(self.config, field, None):
                    return False
            
            # 检查模型列表
            if not self.config.models:
                return False
            
            # 检查默认模型是否在列表中
            if self.config.default_model not in self.config.models:
                return False
            
            # 验证端点格式
            if not self.config.endpoint.startswith(('http://', 'https://')):
                return False
            
            return True
            
        except Exception:
            return False
    
    async def get_available_models(self) -> List[LLMModel]:
        """获取可用模型列表"""
        models = []
        for model_id in self.config.models:
            if model_id in self._model_info:
                models.append(self._model_info[model_id])
            else:
                # 为未知模型创建基本信息
                models.append(LLMModel(
                    id=model_id,
                    name=model_id,
                    provider=LLMProviderType.AZURE_OPENAI,
                    supports_structured_output=True,
                    description=f"Azure OpenAI model: {model_id}"
                ))
        
        return models
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """获取API速率限制信息"""
        return {
            "requests_per_minute": 120,
            "requests_per_second": 2,
            "tokens_per_minute": 120000,
            "concurrent_requests": 10
        }
    
    async def list_deployments(self) -> List[Dict[str, Any]]:
        """列出Azure OpenAI部署
        
        Returns:
            List[Dict[str, Any]]: 部署信息列表
        """
        try:
            # 注意：这需要Azure管理API权限
            # 这里提供一个基本实现框架
            deployments = []
            
            # 基于配置的模型列表返回部署信息
            for model in self.config.models:
                deployments.append({
                    "id": model,
                    "model": model,
                    "status": "succeeded",
                    "created_at": None,
                    "updated_at": None
                })
            
            return deployments
            
        except Exception as e:
            print(f"Failed to list Azure OpenAI deployments: {e}")
            return []


# 注册Azure OpenAI提供商
LLMProviderRegistry.register(LLMProviderType.AZURE_OPENAI, AzureOpenAIProvider)