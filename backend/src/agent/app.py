# mypy: disable - error - code = "no-untyped-def,misc"
import pathlib
from fastapi import FastAPI, Response, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any
import os
from dotenv import load_dotenv

# 导入LLM相关模块
from agent.llm_factory import LLMProviderFactory
from agent.llm_types import LLMProviderType
from agent.configuration import Configuration

# 加载环境变量
backend_dir = pathlib.Path(__file__).parent.parent.parent
env_path = backend_dir / '.env'
load_dotenv(dotenv_path=env_path)

# Define the FastAPI app
app = FastAPI()

# 添加CORS中间件支持前端调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # 开发环境前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/llm-providers")
async def get_llm_providers() -> Dict[str, Any]:
    """获取已配置的LLM提供商和模型列表 - 只返回配置完整的提供商"""
    try:
        factory = LLMProviderFactory()
        providers_info = {}
        
        # 遍历所有提供商类型，但只包含已正确配置的
        for provider_type in LLMProviderType:
            try:
                # 检查提供商是否可用（环境变量是否配置）
                if factory.is_provider_available(provider_type):
                    # 获取提供商实例
                    provider = factory.create_provider(provider_type)
                    
                    # 获取模型列表（现在直接从配置读取，不进行网络调用）
                    models = await provider.get_available_models()
                    
                    # 只包含有模型的提供商
                    if models:
                        providers_info[provider_type.value] = {
                            "name": provider_type.value,
                            "display_name": _get_provider_display_name(provider_type),
                            "available": True,
                            "models": [
                                {
                                    "id": model.id,
                                    "name": model.name,
                                    "description": model.description,
                                    "context_length": model.max_tokens,
                                    "supports_structured_output": model.supports_structured_output
                                }
                                for model in models
                            ]
                        }
                # 忽略未配置的提供商，不将其包含在响应中
            except Exception as e:
                # 忽略配置错误的提供商，不将其包含在响应中
                print(f"Skipping provider {provider_type} due to error: {e}")
        
        # 确保至少有一个提供商可用
        if not providers_info:
            raise HTTPException(status_code=500, detail="No LLM providers are properly configured")
        
        # 获取默认提供商，如果默认提供商不在可用列表中，使用第一个可用的
        default_provider = _get_default_provider()
        if default_provider not in providers_info:
            default_provider = next(iter(providers_info.keys()))
        
        return {
            "providers": providers_info,
            "default_provider": default_provider
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get LLM providers: {str(e)}")


@app.get("/api/llm-providers/{provider_name}/models")
async def get_provider_models(provider_name: str) -> Dict[str, Any]:
    """获取特定提供商的模型列表"""
    try:
        # 验证提供商名称
        try:
            provider_type = LLMProviderType(provider_name)
        except ValueError:
            raise HTTPException(status_code=404, detail=f"Provider '{provider_name}' not found")
        
        factory = LLMProviderFactory()
        
        # 检查提供商是否可用
        if not factory.is_provider_available(provider_type):
            raise HTTPException(status_code=400, detail=f"Provider '{provider_name}' is not available")
        
        # 获取提供商实例和模型列表
        provider = factory.create_provider(provider_type)
        models = await provider.get_available_models()
        
        return {
            "provider": provider_name,
            "models": [
                {
                    "id": model.id,
                    "name": model.name,
                    "description": model.description,
                    "context_length": model.max_tokens,
                    "supports_structured_output": model.supports_structured_output
                }
                for model in models
            ]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get models for provider '{provider_name}': {str(e)}")


@app.get("/api/default-config")
async def get_default_config() -> Dict[str, Any]:
    """获取默认配置，包括默认的LLM provider、model和搜索provider"""
    try:
        from agent.configuration import ModelConfiguration
        
        factory = LLMProviderFactory()
        
        # 获取默认提供商
        default_provider = _get_default_provider()
        
        # 获取默认模型
        default_models = ModelConfiguration.get_default_models_by_provider()
        default_provider_type = LLMProviderType(default_provider)
        default_model = default_models.get(default_provider_type, "")
        
        # 如果没有找到默认模型，尝试获取该提供商的第一个可用模型
        if not default_model and factory.is_provider_available(default_provider_type):
            try:
                provider = factory.create_provider(default_provider_type)
                models = await provider.get_available_models()
                if models:
                    default_model = models[0].id
            except Exception:
                pass
        
        # 获取默认搜索provider（从环境变量）
        default_search_provider = os.getenv("SEARCH_PROVIDER", "google")
        
        # 获取默认努力程度
        default_effort = "medium"  # 固定默认值
        
        return {
            "llm_provider": default_provider,
            "model": default_model,
            "search_provider": default_search_provider,
            "effort": default_effort
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get default config: {str(e)}")


def _get_provider_display_name(provider_type: LLMProviderType) -> str:
    """获取提供商的显示名称"""
    display_names = {
        LLMProviderType.GEMINI: "Google Gemini",
        LLMProviderType.AZURE_OPENAI: "Azure OpenAI",
        LLMProviderType.AWS_BEDROCK: "AWS Bedrock",
        LLMProviderType.OPENAI_COMPATIBLE: "OpenAI Compatible"
    }
    return display_names.get(provider_type, provider_type.value)


def _get_default_provider() -> str:
    """获取默认提供商"""
    factory = LLMProviderFactory()
    
    # 首先尝试使用环境变量中指定的默认提供商
    try:
        default_type = factory.get_default_provider_type()
        if factory.is_provider_available(default_type):
            return default_type.value
    except Exception:
        pass
    
    # 如果默认提供商不可用，按优先级检查可用的提供商
    priority_order = [
        LLMProviderType.GEMINI,
        LLMProviderType.AZURE_OPENAI,
        LLMProviderType.AWS_BEDROCK,
        LLMProviderType.OPENAI_COMPATIBLE
    ]
    
    for provider_type in priority_order:
        if factory.is_provider_available(provider_type):
            return provider_type.value
    
    # 如果没有可用的提供商，返回Gemini作为默认值
    return LLMProviderType.GEMINI.value


def create_frontend_router(build_dir="../frontend/dist"):
    """Creates a router to serve the React frontend.

    Args:
        build_dir: Path to the React build directory relative to this file.

    Returns:
        A Starlette application serving the frontend.
    """
    build_path = pathlib.Path(__file__).parent.parent.parent / build_dir

    if not build_path.is_dir() or not (build_path / "index.html").is_file():
        print(
            f"WARN: Frontend build directory not found or incomplete at {build_path}. Serving frontend will likely fail."
        )
        # Return a dummy router if build isn't ready
        from starlette.routing import Route

        async def dummy_frontend(request):
            return Response(
                "Frontend not built. Run 'npm run build' in the frontend directory.",
                media_type="text/plain",
                status_code=503,
            )

        return Route("/{path:path}", endpoint=dummy_frontend)

    return StaticFiles(directory=build_path, html=True)


# Mount the frontend under /app to not conflict with the LangGraph API routes
app.mount(
    "/app",
    create_frontend_router(),
    name="frontend",
)
