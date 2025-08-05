"""AWS Bedrock LLM提供商实现

提供AWS Bedrock服务的标准化LLM接口。
"""

import os
import json
from typing import Dict, Any, List, Optional, Type
from langchain_aws import ChatBedrock
from langchain_core.language_models import BaseLanguageModel
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

from ..llm_providers import BaseLLMProvider, LLMProviderRegistry
from ..llm_types import (
    LLMRequest,
    LLMResponse,
    LLMProviderType,
    LLMModel,
    BedrockProviderConfig,
    LLMProviderError,
    LLMProviderAPIError,
    LLMProviderTimeoutError
)


class BedrockLLMProvider(BaseLLMProvider):
    """AWS Bedrock LLM提供商
    
    封装AWS Bedrock API的调用，提供标准化接口。
    """
    
    def __init__(self, config: BedrockProviderConfig):
        """初始化Bedrock提供商
        
        Args:
            config: Bedrock配置
        """
        super().__init__(config)
        self.config: BedrockProviderConfig = config
        
        # 初始化AWS Bedrock客户端
        self._bedrock_client = boto3.client(
            'bedrock-runtime',
            region_name=config.region,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key
        )
        
        # 预定义的模型信息
        self._model_info = {
            "anthropic.claude-3-sonnet-20240229-v1:0": LLMModel(
                id="anthropic.claude-3-sonnet-20240229-v1:0",
                name="Claude 3 Sonnet",
                provider=LLMProviderType.AWS_BEDROCK,
                max_tokens=200000,
                supports_structured_output=True,
                description="Balanced model for a wide range of tasks"
            ),
            "anthropic.claude-3-haiku-20240307-v1:0": LLMModel(
                id="anthropic.claude-3-haiku-20240307-v1:0",
                name="Claude 3 Haiku",
                provider=LLMProviderType.AWS_BEDROCK,
                max_tokens=200000,
                supports_structured_output=True,
                description="Fast and efficient model for simple tasks"
            ),
            "anthropic.claude-3-opus-20240229-v1:0": LLMModel(
                id="anthropic.claude-3-opus-20240229-v1:0",
                name="Claude 3 Opus",
                provider=LLMProviderType.AWS_BEDROCK,
                max_tokens=200000,
                supports_structured_output=True,
                description="Most capable model for complex reasoning"
            ),
            "amazon.titan-text-express-v1": LLMModel(
                id="amazon.titan-text-express-v1",
                name="Titan Text Express",
                provider=LLMProviderType.AWS_BEDROCK,
                max_tokens=8000,
                supports_structured_output=False,
                description="Amazon's text generation model"
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
                usage = result.response_metadata.get('usage', {})
            
            return self._create_response(
                content=content,
                model=request.model,
                usage=usage,
                metadata={
                    "task_type": request.task_type,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "region": self.config.region
                }
            )
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            raise LLMProviderAPIError(f"Bedrock API error [{error_code}]: {error_message}")
        except Exception as e:
            raise LLMProviderAPIError(f"Bedrock error: {str(e)}")
    
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
            # 检查模型是否支持结构化输出
            model_info = self._model_info.get(request.model)
            if model_info and not model_info.supports_structured_output:
                # 对于不支持结构化输出的模型，使用提示工程
                schema_prompt = f"\n\nPlease respond in the following JSON format:\n{output_schema.model_json_schema()}"
                request.prompt += schema_prompt
            
            # 创建支持结构化输出的LLM实例
            llm = self.get_langchain_llm(
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # 使用结构化输出（如果支持）
            if model_info and model_info.supports_structured_output:
                structured_llm = llm.with_structured_output(output_schema)
                result = structured_llm.invoke(request.prompt)
            else:
                # 对于不支持的模型，尝试解析JSON响应
                result = llm.invoke(request.prompt)
                content = result.content if hasattr(result, 'content') else str(result)
                try:
                    # 尝试解析JSON
                    import json
                    json_data = json.loads(content)
                    result = output_schema(**json_data)
                except (json.JSONDecodeError, ValueError):
                    # 如果解析失败，返回原始内容
                    result = content
            
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
                    "region": self.config.region
                }
            )
            
        except Exception as e:
            raise LLMProviderAPIError(f"Bedrock structured output error: {str(e)}")
    
    def get_langchain_llm(self, model: str, **kwargs) -> BaseLanguageModel:
        """获取LangChain兼容的LLM实例
        
        Args:
            model: 模型名称
            **kwargs: 额外参数
            
        Returns:
            BaseLanguageModel: LangChain兼容的LLM实例
        """
        return ChatBedrock(
            model_id=model,
            region_name=self.config.region,
            credentials_profile_name=None,  # 使用显式凭证
            client=self._bedrock_client,
            **kwargs
        )
    
    def get_provider_type(self) -> LLMProviderType:
        """返回提供商类型"""
        return LLMProviderType.AWS_BEDROCK
    
    def validate_config(self) -> bool:
        """验证提供商配置是否有效"""
        try:
            # 检查必需的配置项
            required_fields = ['region', 'access_key_id', 'secret_access_key']
            for field in required_fields:
                if not getattr(self.config, field, None):
                    return False
            
            # 检查模型列表
            if not self.config.models:
                return False
            
            # 检查默认模型是否在列表中
            if self.config.default_model not in self.config.models:
                return False
            
            # 验证AWS凭证
            try:
                sts_client = boto3.client(
                    'sts',
                    region_name=self.config.region,
                    aws_access_key_id=self.config.access_key_id,
                    aws_secret_access_key=self.config.secret_access_key
                )
                sts_client.get_caller_identity()
                return True
            except (ClientError, NoCredentialsError):
                return False
            
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
                    provider=LLMProviderType.AWS_BEDROCK,
                    supports_structured_output=True,
                    description=f"AWS Bedrock model: {model_id}"
                ))
        
        return models
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """获取API速率限制信息"""
        return {
            "requests_per_minute": 1000,
            "requests_per_second": 10,
            "tokens_per_minute": 200000,
            "concurrent_requests": 20
        }
    
    async def list_foundation_models(self) -> List[Dict[str, Any]]:
        """列出可用的基础模型
        
        Returns:
            List[Dict[str, Any]]: 基础模型信息列表
        """
        try:
            bedrock_client = boto3.client(
                'bedrock',
                region_name=self.config.region,
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key
            )
            
            response = bedrock_client.list_foundation_models()
            return response.get('modelSummaries', [])
            
        except Exception as e:
            print(f"Failed to list Bedrock foundation models: {e}")
            return []


# 注册Bedrock提供商
LLMProviderRegistry.register(LLMProviderType.AWS_BEDROCK, BedrockLLMProvider)