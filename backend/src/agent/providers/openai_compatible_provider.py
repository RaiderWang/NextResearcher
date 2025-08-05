"""OpenAI兼容LLM提供商实现

提供通用OpenAI兼容API的标准化LLM接口，支持各种OpenAI兼容的服务。
"""

import os
from typing import Dict, Any, List, Optional, Type
from langchain_openai import ChatOpenAI
from langchain_core.language_models import BaseLanguageModel
import openai

from ..llm_providers import BaseLLMProvider, LLMProviderRegistry
from ..llm_types import (
    LLMRequest,
    LLMResponse,
    LLMProviderType,
    LLMModel,
    OpenAICompatibleProviderConfig,
    LLMProviderError,
    LLMProviderAPIError,
    LLMProviderTimeoutError
)


class OpenAICompatibleProvider(BaseLLMProvider):
    """OpenAI兼容LLM提供商
    
    封装OpenAI兼容API的调用，支持各种兼容服务如Ollama、vLLM、LocalAI等。
    """
    
    def __init__(self, config: OpenAICompatibleProviderConfig):
        """初始化OpenAI兼容提供商
        
        Args:
            config: OpenAI兼容配置
        """
        super().__init__(config)
        self.config: OpenAICompatibleProviderConfig = config
        
        # 初始化OpenAI客户端
        self._client = openai.OpenAI(
            api_key=config.api_key,
            base_url=config.base_url
        )
        
        # 动态模型信息（将在运行时获取）
        self._model_info_cache: Dict[str, LLMModel] = {}
    
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
                    "base_url": self.config.base_url
                }
            )
            
        except Exception as e:
            raise LLMProviderAPIError(f"OpenAI Compatible API error: {str(e)}")
    
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
            
            # 尝试使用结构化输出
            try:
                structured_llm = llm.with_structured_output(output_schema)
                result = structured_llm.invoke(request.prompt)
            except Exception:
                # 如果结构化输出失败，使用提示工程
                schema_prompt = f"\n\nPlease respond in the following JSON format:\n{output_schema.model_json_schema()}"
                enhanced_prompt = request.prompt + schema_prompt
                
                response = llm.invoke(enhanced_prompt)
                content = response.content if hasattr(response, 'content') else str(response)
                
                # 尝试解析JSON
                try:
                    import json
                    # 提取JSON部分
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    if json_start != -1 and json_end > json_start:
                        json_str = content[json_start:json_end]
                        json_data = json.loads(json_str)
                        result = output_schema(**json_data)
                    else:
                        result = content
                except (json.JSONDecodeError, ValueError):
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
                    "base_url": self.config.base_url
                }
            )
            
        except Exception as e:
            raise LLMProviderAPIError(f"OpenAI Compatible structured output error: {str(e)}")
    
    def get_langchain_llm(self, model: str, **kwargs) -> BaseLanguageModel:
        """获取LangChain兼容的LLM实例
        
        Args:
            model: 模型名称
            **kwargs: 额外参数
            
        Returns:
            BaseLanguageModel: LangChain兼容的LLM实例
        """
        return ChatOpenAI(
            model=model,
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
            **kwargs
        )
    
    def get_provider_type(self) -> LLMProviderType:
        """返回提供商类型"""
        return LLMProviderType.OPENAI_COMPATIBLE
    
    def validate_config(self) -> bool:
        """验证提供商配置是否有效"""
        try:
            # 检查必需的配置项
            required_fields = ['api_key', 'base_url']
            for field in required_fields:
                if not getattr(self.config, field, None):
                    return False
            
            # 检查模型列表
            if not self.config.models:
                return False
            
            # 检查默认模型是否在列表中
            if self.config.default_model not in self.config.models:
                return False
            
            # 验证URL格式
            if not self.config.base_url.startswith(('http://', 'https://')):
                return False
            
            return True
            
        except Exception:
            return False
    
    async def get_available_models(self) -> List[LLMModel]:
        """获取可用模型列表 - 直接从配置读取，避免API调用"""
        models = []
        
        # 直接使用配置中的模型列表，避免网络调用
        for model_id in self.config.models:
            if model_id in self._model_info_cache:
                models.append(self._model_info_cache[model_id])
            else:
                model = LLMModel(
                    id=model_id,
                    name=model_id,
                    provider=LLMProviderType.OPENAI_COMPATIBLE,
                    supports_structured_output=True,
                    description=f"OpenAI Compatible model: {model_id}"
                )
                models.append(model)
                self._model_info_cache[model_id] = model
        
        return models
    
    async def _fetch_models_from_api(self) -> List[Dict[str, Any]]:
        """从API获取模型列表
        
        Returns:
            List[Dict[str, Any]]: 模型信息列表
        """
        try:
            response = self._client.models.list()
            return [model.dict() if hasattr(model, 'dict') else model for model in response.data]
        except Exception as e:
            print(f"Failed to fetch models from OpenAI Compatible API: {e}")
            return []
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """获取API速率限制信息"""
        # 通用的保守限制，实际限制取决于具体的服务提供商
        return {
            "requests_per_minute": 60,
            "requests_per_second": 1,
            "tokens_per_minute": 60000,
            "concurrent_requests": 5
        }
    
    def get_service_info(self) -> Dict[str, Any]:
        """获取服务信息
        
        Returns:
            Dict[str, Any]: 服务信息
        """
        return {
            "base_url": self.config.base_url,
            "provider_type": self.get_provider_type(),
            "models_count": len(self.config.models),
            "default_model": self.config.default_model
        }


# 注册OpenAI兼容提供商
LLMProviderRegistry.register(LLMProviderType.OPENAI_COMPATIBLE, OpenAICompatibleProvider)