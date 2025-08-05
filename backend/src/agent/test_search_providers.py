#!/usr/bin/env python3
"""
搜索提供商接口测试脚本

此脚本用于测试新的搜索提供商接口和实现。
"""

import os
import asyncio
import sys
from dotenv import load_dotenv

# 添加当前模块路径
sys.path.insert(0, os.path.dirname(__file__))

from search_providers import SearchRequest
from search_factory import SearchProviderFactory
from providers.google_search_provider import GoogleSearchProvider
from providers.tavily_search_provider import TavilySearchProvider

# 加载环境变量 - 从backend/.env文件读取
# 找到backend目录的路径
backend_dir = os.path.join(os.path.dirname(__file__), '..', '..')
env_path = os.path.join(backend_dir, '.env')
load_dotenv(dotenv_path=env_path)


async def test_search_provider_interface():
    """测试搜索提供商接口"""
    print("🔍 测试搜索提供商接口...")
    
    try:
        # 1. 测试工厂方法
        print("\n1. 测试搜索提供商工厂:")
        available_providers = SearchProviderFactory.get_available_providers()
        print(f"   可用提供商: {available_providers}")
        
        # 2. 测试提供商信息
        print("\n2. 测试提供商信息:")
        for provider_name in available_providers:
            try:
                info = SearchProviderFactory.get_provider_info(provider_name)
                print(f"   {provider_name}: {info['description']}")
                print(f"   必需配置: {info.get('required_config_keys', [])}")
                print(f"   速率限制: {info.get('rate_limits', {})}")
                if 'supported_features' in info:
                    print(f"   支持功能: {info['supported_features']}")
            except Exception as e:
                print(f"   {provider_name}: 获取信息失败 - {e}")
        
        # 3. 测试Google搜索提供商
        print("\n3. 测试Google搜索提供商:")
        if os.getenv("GEMINI_API_KEY"):
            try:
                # 创建搜索提供商
                google_provider = SearchProviderFactory.create_provider("google")
                print(f"   ✅ 创建Google提供商成功: {google_provider.get_provider_name()}")
                
                # 创建搜索请求
                search_request = SearchRequest(
                    query="Python编程语言最新版本",
                    max_results=3,
                    language="zh-CN"
                )
                print(f"   📝 搜索查询: {search_request.query}")
                
                # 执行搜索
                print("   🔄 执行搜索...")
                result = await google_provider.search(search_request)
                
                # 显示结果
                print(f"   📊 搜索结果:")
                print(f"   内容长度: {len(result.content)} 字符")
                print(f"   来源数量: {len(result.sources)}")
                
                if result.sources:
                    print("   前3个来源:")
                    for i, source in enumerate(result.sources[:3]):
                        print(f"     {i+1}. {source['title'][:50]}...")
                        print(f"        URL: {source['url']}")
                
                # 显示性能指标
                metrics = google_provider.get_search_metrics()
                if metrics:
                    print(f"   ⏱️  性能指标:")
                    print(f"   搜索耗时: {metrics.search_time:.2f} 秒")
                    print(f"   结果数量: {metrics.result_count}")
                    print(f"   API调用: {metrics.api_calls_used}")
                
                print("   ✅ Google搜索测试成功!")
                
            except Exception as e:
                print(f"   ❌ Google搜索测试失败: {e}")
        else:
            print("   ⚠️  跳过Google搜索测试（未设置GEMINI_API_KEY）")
        
        # 测试Tavily搜索提供商
        print("\n3.2 测试Tavily搜索提供商:")
        if os.getenv("TAVILY_API_KEY"):
            try:
                # 创建搜索提供商
                tavily_provider = SearchProviderFactory.create_provider("tavily")
                print(f"   ✅ 创建Tavily提供商成功: {tavily_provider.get_provider_name()}")
                
                # 创建搜索请求
                search_request = SearchRequest(
                    query="人工智能最新发展趋势",
                    max_results=3,
                    language="zh-CN"
                )
                print(f"   📝 搜索查询: {search_request.query}")
                
                # 执行搜索
                print("   🔄 执行搜索...")
                result = await tavily_provider.search(search_request)
                
                # 显示结果
                print(f"   📊 搜索结果:")
                print(f"   内容长度: {len(result.content)} 字符")
                print(f"   来源数量: {len(result.sources)}")
                
                if result.sources:
                    print("   前3个来源:")
                    for i, source in enumerate(result.sources[:3]):
                        print(f"     {i+1}. {source['title'][:50]}...")
                        print(f"        URL: {source['url']}")
                
                # 显示性能指标
                metrics = tavily_provider.get_search_metrics()
                if metrics:
                    print(f"   ⏱️  性能指标:")
                    print(f"   搜索耗时: {metrics.search_time:.2f} 秒")
                    print(f"   结果数量: {metrics.result_count}")
                    print(f"   API调用: {metrics.api_calls_used}")
                
                print("   ✅ Tavily搜索测试成功!")
                
            except Exception as e:
                print(f"   ❌ Tavily搜索测试失败: {e}")
        else:
            print("   ⚠️  跳过Tavily搜索测试（未设置TAVILY_API_KEY）")
        
        # 4. 测试配置验证
        print("\n4. 测试配置验证:")
        
        # 测试Google配置验证
        test_config = {"api_key": "test_key"}
        is_valid, message = SearchProviderFactory.validate_provider_config("google", test_config)
        print(f"   Google配置有效性: {is_valid} - {message}")
        
        # 测试Tavily配置验证
        tavily_test_config = {"api_key": "tvly-test_api_key_with_sufficient_length"}
        is_valid, message = SearchProviderFactory.validate_provider_config("tavily", tavily_test_config)
        print(f"   Tavily配置有效性: {is_valid} - {message}")
        
        print("\n🎉 所有测试完成!")
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """主函数"""
    print("=" * 60)
    print("搜索提供商接口测试")
    print("=" * 60)
    
    await test_search_provider_interface()


if __name__ == "__main__":
    asyncio.run(main()) 