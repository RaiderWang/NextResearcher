"""LLM提供商自动注册模块

自动导入并注册所有可用的LLM提供商。
"""

from .llm_types import LLMProviderType
from .llm_providers import LLMProviderRegistry

# 导入所有提供商实现
try:
    from .providers.gemini_llm_provider import GeminiLLMProvider
    LLMProviderRegistry.register(LLMProviderType.GEMINI, GeminiLLMProvider)
    print("✅ Registered Gemini LLM Provider")
except ImportError as e:
    print(f"⚠️  Failed to register Gemini LLM Provider: {e}")

try:
    from .providers.azure_openai_provider import AzureOpenAIProvider
    LLMProviderRegistry.register(LLMProviderType.AZURE_OPENAI, AzureOpenAIProvider)
    print("✅ Registered Azure OpenAI Provider")
except ImportError as e:
    print(f"⚠️  Failed to register Azure OpenAI Provider: {e}")

try:
    from .providers.bedrock_llm_provider import BedrockLLMProvider
    LLMProviderRegistry.register(LLMProviderType.AWS_BEDROCK, BedrockLLMProvider)
    print("✅ Registered AWS Bedrock Provider")
except ImportError as e:
    print(f"⚠️  Failed to register AWS Bedrock Provider: {e}")

try:
    from .providers.openai_compatible_provider import OpenAICompatibleProvider
    LLMProviderRegistry.register(LLMProviderType.OPENAI_COMPATIBLE, OpenAICompatibleProvider)
    print("✅ Registered OpenAI Compatible Provider")
except ImportError as e:
    print(f"⚠️  Failed to register OpenAI Compatible Provider: {e}")

# 显示注册的提供商
registered_providers = LLMProviderRegistry.get_available_types()
print(f"🔧 Total registered providers: {len(registered_providers)}")
print(f"📋 Available providers: {[p.value for p in registered_providers]}")