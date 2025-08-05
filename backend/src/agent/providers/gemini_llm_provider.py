"""Google Gemini LLM提供商实现

基于现有的Gemini集成，提供标准化的LLM接口。
"""

import os
from typing import Dict, Any, List, Optional, Type
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseLanguageModel
from google.genai import Client

from ..llm_providers import BaseLLMProvider, LLMProviderRegistry
from ..llm_types import (
    LLMRequest,
    LLMResponse,
    LLMProviderType,
    LLMModel,
    GeminiProviderConfig,
    LLMProviderError,
    LLMProviderAPIError,
    LLMProviderTimeoutError
)


class GeminiLLMProvider(BaseLLMProvider):
    """Google Gemini LLM提供商
    
    封装Google Gemini API的调用，提供标准化接口。
    """
    
    def __init__(self, config: GeminiProviderConfig):
        """初始化Gemini提供商
        
        Args:
            config: Gemini配置
        """
        super().__init__(config)
        self.config: GeminiProviderConfig = config
        self._genai_client = Client(api_key=config.api_key)
        
        # 预定义的模型信息
        self._model_info = {
            "gemini-2.0-flash": LLMModel(
                id="gemini-2.0-flash",
                name="Gemini 2.0 Flash",
                provider=LLMProviderType.GEMINI,
                max_tokens=8192,
                supports_structured_output=True,
                description="Fast and efficient model for general tasks"
            ),
            "gemini-2.5-flash": LLMModel(
                id="gemini-2.5-flash",
                name="Gemini 2.5 Flash",
                provider=LLMProviderType.GEMINI,
                max_tokens=8192,
                supports_structured_output=True,
                description="Enhanced flash model with improved capabilities"
            ),
            "gemini-2.5-pro": LLMModel(
                id="gemini-2.5-pro",
                name="Gemini 2.5 Pro",
                provider=LLMProviderType.GEMINI,
                max_tokens=32768,
                supports_structured_output=True,
                description="Advanced model for complex reasoning tasks"
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
            
            return self._create_response(
                content=content,
                model=request.model,
                metadata={
                    "task_type": request.task_type,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens
                }
            )
            
        except Exception as e:
            raise LLMProviderAPIError(f"Gemini API error: {str(e)}")
    
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
            print(f"🔧 Gemini structured_output - 开始生成")
            print(f"🔧 - 模型: {request.model}")
            print(f"🔧 - schema: {output_schema.__name__}")
            print(f"🔧 - prompt长度: {len(request.prompt)}")
            
            # 创建支持结构化输出的LLM实例
            llm = self.get_langchain_llm(
                model=request.model,
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            
            # 使用结构化输出
            structured_llm = llm.with_structured_output(output_schema)
            print(f"🔧 Gemini - 调用structured_llm.invoke")
            
            try:
                result = structured_llm.invoke(request.prompt)
                print(f"🔧 Gemini - 结构化输出结果: {result}")
                print(f"🔧 Gemini - 结果类型: {type(result)}")
                
                # 如果结果为None，尝试使用普通生成然后解析
                if result is None:
                    print(f"⚠️ Gemini - 结构化输出为None，尝试普通生成")
                    plain_result = llm.invoke(request.prompt)
                    print(f"🔧 Gemini - 普通生成结果: {plain_result}")
                    
                    # 检查是否是MAX_TOKENS问题
                    if (hasattr(plain_result, 'response_metadata') and 
                        plain_result.response_metadata.get('finish_reason') == 'MAX_TOKENS'):
                        print(f"⚠️ Gemini - 检测到MAX_TOKENS问题，使用更大的token限制重试")
                        
                        # 增加token限制重试
                        retry_llm = self.get_langchain_llm(
                            model=request.model,
                            temperature=request.temperature,
                            max_tokens=min(request.max_tokens * 2, 8000)  # 翻倍但不超过8000
                        )
                        retry_structured_llm = retry_llm.with_structured_output(output_schema)
                        result = retry_structured_llm.invoke(request.prompt)
                        print(f"🔧 Gemini - 重试结果: {result}")
                    
                    # 如果仍然为None，尝试手动解析
                    if result is None:
                        # 尝试解析为结构化数据
                        try:
                            import json
                            content_text = plain_result.content if hasattr(plain_result, 'content') else str(plain_result)
                            # 简单的JSON解析尝试
                            if content_text.strip().startswith('{'):
                                parsed_data = json.loads(content_text.strip())
                                result = output_schema(**parsed_data)
                                print(f"🔧 Gemini - 手动解析成功: {result}")
                        except Exception as parse_error:
                            print(f"⚠️ Gemini - 手动解析失败: {parse_error}")
                            # 使用原始文本作为content，但structured_data为None
                            content = content_text
                            result = None
                
                # 生成文本表示
                content = str(result) if result else ""
                print(f"🔧 Gemini - 生成的content: '{content}'")
                
            except Exception as invoke_error:
                print(f"❌ Gemini - invoke调用失败: {invoke_error}")
                raise invoke_error
            
            return self._create_response(
                content=content,
                model=request.model,
                structured_data=result,
                metadata={
                    "task_type": request.task_type,
                    "temperature": request.temperature,
                    "max_tokens": request.max_tokens,
                    "structured_output": True,
                    "schema": output_schema.__name__
                }
            )
            
        except Exception as e:
            raise LLMProviderAPIError(f"Gemini structured output error: {str(e)}")
    
    def get_langchain_llm(self, model: str, **kwargs) -> BaseLanguageModel:
        """获取LangChain兼容的LLM实例
        
        Args:
            model: 模型名称
            **kwargs: 额外参数
            
        Returns:
            BaseLanguageModel: LangChain兼容的LLM实例
        """
        return ChatGoogleGenerativeAI(
            model=model,
            api_key=self.config.api_key,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
            **kwargs
        )
    
    def get_provider_type(self) -> LLMProviderType:
        """返回提供商类型"""
        return LLMProviderType.GEMINI
    
    def validate_config(self) -> bool:
        """验证提供商配置是否有效"""
        try:
            # 检查API密钥
            if not self.config.api_key:
                return False
            
            # 检查模型列表
            if not self.config.models:
                return False
            
            # 检查默认模型是否在列表中
            if self.config.default_model not in self.config.models:
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
                    provider=LLMProviderType.GEMINI,
                    supports_structured_output=True,
                    description=f"Gemini model: {model_id}"
                ))
        
        return models
    
    def get_rate_limits(self) -> Dict[str, Any]:
        """获取API速率限制信息"""
        return {
            "requests_per_minute": 60,
            "requests_per_second": 1,
            "tokens_per_minute": 32000,
            "concurrent_requests": 5
        }
    
    async def generate_with_google_search(self, prompt: str, model: str) -> tuple[str, List[Dict[str, str]]]:
        """使用Google搜索功能生成内容（保持与原有代码兼容）
        
        Args:
            prompt: 输入提示词
            model: 使用的模型
            
        Returns:
            tuple[str, List[Dict[str, str]]]: (生成的内容, 搜索来源列表)
        """
        try:
            # 使用Google GenAI客户端进行搜索增强生成
            response = self._genai_client.models.generate_content(
                model=model,
                contents=prompt,
                config={
                    "tools": [{"google_search": {}}],
                    "temperature": 0,
                }
            )
            
            # 提取内容和来源
            content = response.text if response.text else ""
            sources = []
            
            # 处理grounding metadata（如果存在）
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    grounding_chunks = candidate.grounding_metadata.grounding_chunks
                    for i, chunk in enumerate(grounding_chunks):
                        if hasattr(chunk, 'web') and chunk.web:
                            sources.append({
                                "label": f"[{i+1}]",
                                "url": chunk.web.uri,
                                "title": chunk.web.title or "Unknown Title"
                            })
            
            return content, sources
            
        except Exception as e:
            raise LLMProviderAPIError(f"Gemini Google Search error: {str(e)}")


# 注册Gemini提供商
LLMProviderRegistry.register(LLMProviderType.GEMINI, GeminiLLMProvider)