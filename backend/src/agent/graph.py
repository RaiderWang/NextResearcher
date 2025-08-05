import os

from agent.tools_and_schemas import SearchQueryList, Reflection
from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Send
from langgraph.graph import StateGraph
from langgraph.graph import START, END
from langchain_core.runnables import RunnableConfig
from google.genai import Client

# 导入搜索提供商相关模块
from agent.search_factory import SearchProviderFactory
from agent.search_providers import SearchRequest

# 导入LLM服务相关模块
# 确保提供商注册
from agent import provider_registry
from agent.llm_service import get_llm_service, ConfigurableLLMService
from agent.llm_types import LLMRequest, LLMTaskType

from agent.state import (
    OverallState,
    QueryGenerationState,
    ReflectionState,
    WebSearchState,
)
from agent.configuration import Configuration
from agent.prompts import (
    get_current_date,
    query_writer_instructions,
    web_searcher_instructions,
    reflection_instructions,
    answer_instructions,
)
from agent.utils import (
    get_research_topic,
)

# 加载环境变量 - 从backend/.env文件读取
backend_dir = os.path.join(os.path.dirname(__file__), '..', '..')
env_path = os.path.join(backend_dir, '.env')
load_dotenv(dotenv_path=env_path)

if os.getenv("GEMINI_API_KEY") is None:
    raise ValueError("GEMINI_API_KEY is not set")

# Used for Google Search API
genai_client = Client(api_key=os.getenv("GEMINI_API_KEY"))


# Nodes
async def generate_query(state: OverallState, config: RunnableConfig) -> QueryGenerationState:
    """LangGraph node that generates search queries based on the User's question.

    Uses the configured LLM provider to create optimized search queries for web research based on
    the User's question.

    Args:
        state: Current graph state containing the User's question
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated queries
    """
    configurable = Configuration.from_runnable_config(config)

    # check for custom initial search query count
    if state.get("initial_search_query_count") is None:
        state["initial_search_query_count"] = configurable.number_of_initial_queries

    try:
        # 添加详细的状态调试信息
        print(f"🐛 DEBUG: 当前状态的所有键: {list(state.keys())}")
        print(f"🐛 DEBUG: 状态中的llm_provider值: {state.get('llm_provider')}")
        print(f"🐛 DEBUG: 状态中的reasoning_model值: {state.get('reasoning_model')}")
        
        # 从状态中获取用户选择的参数
        llm_provider_from_state = state.get("llm_provider")
        reasoning_model_from_state = state.get("reasoning_model")
        
        # 创建修改过的config来传递状态参数给ConfigurableLLMService
        modified_config = config.copy() if config else {}
        if "configurable" not in modified_config:
            modified_config["configurable"] = {}
        
        # 传递llm_provider
        if llm_provider_from_state and llm_provider_from_state.strip():
            modified_config["configurable"]["llm_provider"] = llm_provider_from_state
            print(f"🔧 generate_query - 从状态获取llm_provider: {llm_provider_from_state}")
        
        # 传递reasoning_model
        if reasoning_model_from_state and reasoning_model_from_state.strip():
            modified_config["configurable"]["reasoning_model"] = reasoning_model_from_state
            print(f"🔧 generate_query - 从状态获取reasoning_model: {reasoning_model_from_state}")
        
        llm_service = ConfigurableLLMService(modified_config)
        
        # Format the prompt
        current_date = get_current_date()
        formatted_prompt = query_writer_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state["messages"]),
            number_queries=state["initial_search_query_count"],
        )
        
        # 添加调试日志
        print(f"🔍 generate_query: 准备调用LLM服务")
        print(f"🔍 formatted_prompt长度: {len(formatted_prompt)}")
        
        # 从state中获取用户选择的模型，如果没有则使用配置默认值
        model = state.get("reasoning_model", configurable.query_generator_model)
        print(f"🔍 generate_query - 使用模型: {model}")
        print(f"🔍 generate_query - 模型来源: {'state' if state.get('reasoning_model') else 'config'}")
        
        # 生成搜索查询 - 使用正确的LLMService接口
        result = await llm_service.generate_structured(
            prompt=formatted_prompt,
            output_schema=SearchQueryList,
            model=model,
            task_type=LLMTaskType.QUERY_GENERATION,
            temperature=0.7,  # 降低temperature提高一致性
            max_tokens=2000   # 增加token限制
        )
        
        # 验证结果
        if not result or not hasattr(result, 'query') or not result.query:
            raise ValueError("Failed to generate valid search queries")
            
        return {"search_query": result.query}
    except Exception as e:
        print(f"Error in generate_query: {e}")
        # 返回降级搜索查询
        research_topic = get_research_topic(state["messages"])
        fallback_query = [research_topic] if research_topic else ["general search"]
        return {"search_query": fallback_query}


def continue_to_web_research(state: OverallState):
    """LangGraph node that sends the search queries to the web research node.

    This is used to spawn n number of web research nodes, one for each search query.
    """
    # 安全访问 search_query
    search_queries = state.get("search_query", [])
    if not search_queries:
        # 如果没有搜索查询，提供降级方案
        search_queries = ["general search"]
    
    search_provider = state.get("search_provider", "google")
    
    return [
        Send("web_research", {
            "search_query": search_query, 
            "id": int(idx),
            "search_provider": search_provider
        })
        for idx, search_query in enumerate(search_queries)
    ]


async def web_research(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """LangGraph node that performs web research using configurable search providers.

    Executes a web search using the configured search provider (Google, Tavily, etc.).

    Args:
        state: Current graph state containing the search query and research loop count
        config: Configuration for the runnable, including search provider settings

    Returns:
        Dictionary with state update, including sources_gathered, research_loop_count, and web_research_results
    """
    # 获取配置
    configurable = Configuration.from_runnable_config(config)
    
    # 从状态中获取搜索提供商，如果没有则使用配置中的默认值
    search_provider_name = state.get("search_provider", configurable.search_provider)
    
    try:
        # 创建搜索提供商实例
        print(f"🔍 使用搜索提供商: {search_provider_name}")
        search_provider = SearchProviderFactory.create_provider(
            search_provider_name
        )
        
        # 创建搜索请求
        search_request = SearchRequest(
            query=state["search_query"],
            max_results=configurable.search_results_limit,
            language="zh-CN"
        )
        
        # 执行搜索
        search_result = await search_provider.search(search_request)
        
        # 将搜索结果转换为兼容格式
        sources_gathered = []
        for i, source in enumerate(search_result.sources):
            # 生成简短URL标识符
            short_url = f"[{i+1}]"
            real_url = source.get("url", "")
            title = source.get("title", "")
            
            # 添加详细日志
            print(f"🔗 处理来源{i+1}: URL={real_url[:100]}..." if len(real_url) > 100 else f"🔗 处理来源{i+1}: URL={real_url}")
            print(f"📖 标题: {title}")
            
            sources_gathered.append({
                "label": short_url,
                "short_url": short_url,
                "value": real_url,
                "title": title
            })
        
        print(f"✅ {search_provider_name}搜索完成: 获得{len(sources_gathered)}个来源，内容长度{len(search_result.content)}")
        
        # 添加sources详细日志
        for source in sources_gathered:
            print(f"📚 来源详细信息: {source['short_url']} -> {source['value'][:50]}...")
            print(f"    标题: {source['title']}")
        
        return {
            "sources_gathered": sources_gathered,
            "search_query": [state.get("search_query", "unknown query")],
            "web_research_result": [search_result.content],
        }
        
    except Exception as e:
                 # 如果搜索提供商失败，回退到原始的Google搜索方法
         print(f"Search provider failed, falling back to Google Search: {e}")
         return await _fallback_google_search(state, config)


async def _fallback_google_search(state: WebSearchState, config: RunnableConfig) -> OverallState:
    """原始的Google搜索实现，用作后备方案"""
    # Configure
    configurable = Configuration.from_runnable_config(config)
    formatted_prompt = web_searcher_instructions.format(
        current_date=get_current_date(),
        research_topic=state["search_query"],
    )

    # Uses the google genai client as the langchain client doesn't return grounding metadata
    response = genai_client.models.generate_content(
        model=configurable.query_generator_model,
        contents=formatted_prompt,
        config={
            "tools": [{"google_search": {}}],
            "temperature": 0,
        },
    )
    # 直接从grounding chunks获取真实URL，不使用resolve_urls转换
    grounding_chunks = response.candidates[0].grounding_metadata.grounding_chunks
    
    # 构建来源列表 - 直接使用真实URL
    sources_gathered = []
    for i, chunk in enumerate(grounding_chunks):
        if hasattr(chunk, 'web') and hasattr(chunk.web, 'uri'):
            real_url = chunk.web.uri
            title = getattr(chunk.web, 'title', f"搜索结果{i+1}")
            sources_gathered.append({
                "label": f"来源{i+1}",
                "short_url": f"[{i+1}]",  # 简单的标记
                "value": real_url,  # 使用真实URL
                "title": title
            })
    
    # 清理Gemini API自动生成的假链接
    import re
    
    # 移除所有形如 [label](https://vertexaisearch.cloud.google.com/...) 的假链接
    pattern = r'\[([^\]]+)\]\(https://vertexaisearch\.cloud\.google\.com/[^)]+\)'
    modified_text = re.sub(pattern, r'\1', response.text)
    
    # 移除单独的vertexaisearch链接
    pattern2 = r'https://vertexaisearch\.cloud\.google\.com/[^\s\])]+'
    modified_text = re.sub(pattern2, '', modified_text).strip()

    return {
        "sources_gathered": sources_gathered,
        "search_query": [state.get("search_query", "unknown query")],
        "web_research_result": [modified_text],
    }


async def reflection(state: OverallState, config: RunnableConfig) -> ReflectionState:
    """LangGraph node that identifies knowledge gaps and generates potential follow-up queries.

    Analyzes the current summary to identify areas for further research and generates
    potential follow-up queries. Uses structured output to extract
    the follow-up query in JSON format.

    Args:
        state: Current graph state containing the running summary and research topic
        config: Configuration for the runnable, including LLM provider settings

    Returns:
        Dictionary with state update, including search_query key containing the generated follow-up query
    """
    configurable = Configuration.from_runnable_config(config)
    # Increment the research loop count and get the reasoning model
    state["research_loop_count"] = state.get("research_loop_count", 0) + 1
    reasoning_model = state.get("reasoning_model", configurable.reflection_model)

    try:
        # 从状态中获取用户选择的参数
        llm_provider_from_state = state.get("llm_provider")
        reasoning_model_from_state = state.get("reasoning_model")
        
        # 创建修改过的config来传递状态参数给ConfigurableLLMService
        modified_config = config.copy() if config else {}
        if "configurable" not in modified_config:
            modified_config["configurable"] = {}
        
        # 传递llm_provider
        if llm_provider_from_state and llm_provider_from_state.strip():
            modified_config["configurable"]["llm_provider"] = llm_provider_from_state
            print(f"🔧 reflection - 从状态获取llm_provider: {llm_provider_from_state}")
        
        # 传递reasoning_model
        if reasoning_model_from_state and reasoning_model_from_state.strip():
            modified_config["configurable"]["reasoning_model"] = reasoning_model_from_state
            print(f"🔧 reflection - 从状态获取reasoning_model: {reasoning_model_from_state}")
        
        llm_service = ConfigurableLLMService(modified_config)
        
        # Format the prompt
        current_date = get_current_date()
        formatted_prompt = reflection_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state.get("messages", [])),
            summaries="\n\n---\n\n".join(state.get("web_research_result", [])),
        )
        
        # 添加调试日志
        print(f"🤔 reflection: 准备调用LLM服务")
        print(f"🤔 formatted_prompt长度: {len(formatted_prompt)}")
        print(f"🤔 使用模型: {reasoning_model}")
        
        # 生成反思结果 - 使用正确的LLMService接口，增加token限制
        result = await llm_service.generate_structured(
            prompt=formatted_prompt,
            output_schema=Reflection,
            model=reasoning_model,
            task_type=LLMTaskType.REFLECTION,
            temperature=0.7,  # 降低temperature提高一致性
            max_tokens=4000   # 大幅增加token限制
        )
        
        print(f"🤔 反思结果对象: {result}")
        print(f"🤔 反思结果类型: {type(result)}")
        
        # 正确提取结构化数据
        structured_data = result.structured_data if result else None
        
        if structured_data:
            print(f"🤔 structured_data: {structured_data}")
            print(f"🤔 structured_data 类型: {type(structured_data)}")
            print(f"🤔 is_sufficient: {getattr(structured_data, 'is_sufficient', 'MISSING')}")
            print(f"🤔 knowledge_gap: {getattr(structured_data, 'knowledge_gap', 'MISSING')}")
            print(f"🤔 follow_up_queries: {getattr(structured_data, 'follow_up_queries', 'MISSING')}")
            print(f"🤔 follow_up_queries 类型: {type(getattr(structured_data, 'follow_up_queries', None))}")
        else:
            print("🤔 没有找到 structured_data")
        
        follow_up_queries = getattr(structured_data, 'follow_up_queries', []) if structured_data else []
        print(f"❓ 处理后的后续查询: {follow_up_queries}")

        return {
            "is_sufficient": getattr(structured_data, 'is_sufficient', False) if structured_data else False,
            "knowledge_gap": getattr(structured_data, 'knowledge_gap', "") if structured_data else "",
            "follow_up_queries": follow_up_queries,
            "research_loop_count": state.get("research_loop_count", 0),
            "number_of_ran_queries": len(state.get("search_query", [])),
        }
    except Exception as e:
        print(f"Error in reflection: {e}")
        # 返回默认值，表示研究充分
        return {
            "is_sufficient": True,
            "knowledge_gap": "",
            "follow_up_queries": [],
            "research_loop_count": state.get("research_loop_count", 0),
            "number_of_ran_queries": len(state.get("search_query", [])),
        }


def evaluate_research(
    state: OverallState,
    config: RunnableConfig,
) -> OverallState:
    """LangGraph routing function that determines the next step in the research flow.

    Controls the research loop by deciding whether to continue gathering information
    or to finalize the summary based on the configured maximum number of research loops.

    Args:
        state: Current graph state containing the research loop count
        config: Configuration for the runnable, including max_research_loops setting

    Returns:
        String literal indicating the next node to visit ("web_research" or "finalize_summary")
    """
    configurable = Configuration.from_runnable_config(config)
    max_research_loops = (
        state.get("max_research_loops")
        if state.get("max_research_loops") is not None
        else configurable.max_research_loops
    )
    is_sufficient = state.get("is_sufficient", False)
    research_count = state.get("research_loop_count", 0)
    follow_up_queries = state.get("follow_up_queries", [])
    print(f"🔄 评估研究状态: is_sufficient={is_sufficient}, research_count={research_count}, max_loops={max_research_loops}")
    print(f"🔄 状态中的所有键: {list(state.keys())}")
    print(f"🔄 从状态获取的follow_up_queries: {follow_up_queries}")
    
    if is_sufficient or research_count >= max_research_loops:
        print("✅ 研究充分，前往 finalize_answer")
        return "finalize_answer"
    else:
        print("🔄 继续研究，生成更多查询")
        print(f"📝 生成的后续查询: {follow_up_queries}")
        
        # 如果没有后续查询但还需要研究，强制结束并生成答案
        if not follow_up_queries:
            print("⚠️ 没有后续查询但研究不充分，强制前往 finalize_answer")
            return "finalize_answer"
        
        return [
            Send(
                "web_research",
                {
                    "search_query": follow_up_query,
                    "id": state.get("number_of_ran_queries", 0) + int(idx),
                    "search_provider": state.get("search_provider", "google"),
                },
            )
            for idx, follow_up_query in enumerate(follow_up_queries)
        ]


async def finalize_answer(state: OverallState, config: RunnableConfig):
    """LangGraph node that finalizes the research summary.

    Prepares the final output by deduplicating and formatting sources, then
    combining them with the running summary to create a well-structured
    research report with proper citations.

    Args:
        state: Current graph state containing the running summary and sources gathered

    Returns:
        Dictionary with state update, including running_summary key containing the formatted final summary with sources
    """
    print("🔥 finalize_answer 函数被调用")
    print(f"📊 状态信息: web_research_result数量={len(state.get('web_research_result', []))}")
    print(f"📚 sources_gathered数量={len(state.get('sources_gathered', []))}")
    
    configurable = Configuration.from_runnable_config(config)
    reasoning_model = state.get("reasoning_model") or configurable.answer_model
    print(f"🤖 使用模型: {reasoning_model}")

    try:
        # 从状态中获取用户选择的参数
        llm_provider_from_state = state.get("llm_provider")
        reasoning_model_from_state = state.get("reasoning_model")
        
        # 创建修改过的config来传递状态参数给ConfigurableLLMService
        modified_config = config.copy() if config else {}
        if "configurable" not in modified_config:
            modified_config["configurable"] = {}
        
        # 传递llm_provider
        if llm_provider_from_state and llm_provider_from_state.strip():
            modified_config["configurable"]["llm_provider"] = llm_provider_from_state
            print(f"🔧 finalize_answer - 从状态获取llm_provider: {llm_provider_from_state}")
        
        # 传递reasoning_model
        if reasoning_model_from_state and reasoning_model_from_state.strip():
            modified_config["configurable"]["reasoning_model"] = reasoning_model_from_state
            print(f"🔧 finalize_answer - 从状态获取reasoning_model: {reasoning_model_from_state}")
        
        llm_service = ConfigurableLLMService(modified_config)
        
        # Format the prompt with enhanced summaries including source information
        current_date = get_current_date()
        research_results = state.get("web_research_result", [])
        all_sources = state.get("sources_gathered", [])
        
        # 创建增强的summaries，包含来源信息
        enhanced_summaries = []
        for i, result in enumerate(research_results):
            enhanced_summary = f"## Research Result {i+1}\n\n{result}"
            
            # 添加相关来源信息
            if all_sources:
                # 为每个research result添加对应的来源信息
                sources_for_this_result = []
                start_idx = i * 5  # 假设每个研究结果对应5个来源
                end_idx = min(start_idx + 5, len(all_sources))
                
                for j in range(start_idx, end_idx):
                    if j < len(all_sources):
                        source = all_sources[j]
                        sources_for_this_result.append(f"[{j+1}] {source.get('title', '')} - {source.get('value', '')}")
                
                if sources_for_this_result:
                    enhanced_summary += f"\n\n**Available sources for this section:**\n" + "\n".join(sources_for_this_result)
            
            enhanced_summaries.append(enhanced_summary)
        
        # 如果没有研究结果但有来源，创建一个基本summary
        if not enhanced_summaries and all_sources:
            basic_summary = "## Available Research Sources\n\n"
            for i, source in enumerate(all_sources):
                basic_summary += f"[{i+1}] {source.get('title', '')} - {source.get('value', '')}\n"
            enhanced_summaries.append(basic_summary)
        
        formatted_prompt = answer_instructions.format(
            current_date=current_date,
            research_topic=get_research_topic(state.get("messages", [])),
            summaries="\n---\n\n".join(enhanced_summaries) if enhanced_summaries else "No research results available.",
        )
        
        print(f"📝 增强的summaries包含 {len(enhanced_summaries)} 个部分")
        print(f"🔗 总共 {len(all_sources)} 个可用来源")

        # 添加调试日志
        print(f"🔥 finalize_answer: 准备调用LLM服务")
        print(f"🔥 formatted_prompt长度: {len(formatted_prompt)}")
        print(f"🔥 使用模型: {reasoning_model}")
        
        # 生成最终答案 - 使用正确的LLMService接口
        result = await llm_service.generate(
            prompt=formatted_prompt,
            model=reasoning_model,
            task_type=LLMTaskType.ANSWER_GENERATION,
            temperature=0.3,  # 稍微提高一点creativity
            max_tokens=8000   # 增加token限制确保完整输出
        )

        # 添加详细日志：显示LLM生成的原始内容
        print(f"🤖 LLM生成的原始回复前200字符:")
        print(f"   {result.content[:200] if result.content else 'None'}...")
        
        print(f"📚 状态中的所有来源:")
        for i, source in enumerate(state.get("sources_gathered", [])):
            print(f"   {i+1}. {source.get('short_url')} -> {source.get('value', '')[:50]}...")
            print(f"      标题: {source.get('title', '')}")
        
        # 改进的URL替换逻辑
        import re
        unique_sources = []
        all_sources = state.get("sources_gathered", [])
        
        print(f"🔄 开始URL替换处理，共有 {len(all_sources)} 个来源")
        
        # 策略1: 使用正则表达式查找并替换所有引用格式
        import re
        
        print(f"🔍 开始处理各种引用格式...")
        print(f"📝 原始内容片段: {result.content[:200]}...")
        
        # 使用统一的正则表达式查找所有引用格式（单个和组合）
        citation_pattern = r'\[(\d+(?:\s*,\s*\d+)*)\]'
        
        # 查找所有匹配的引用
        matches = list(re.finditer(citation_pattern, result.content))
        print(f"🔍 发现 {len(matches)} 个引用格式")
        
        # 从后往前替换，避免位置偏移问题
        for match in reversed(matches):
            full_match = match.group(0)  # 完整的匹配，如 [1] 或 [1, 2, 3]
            ref_numbers_str = match.group(1)  # 数字部分，如 "1" 或 "1, 2, 3"
            
            print(f"🔍 处理引用: {full_match} (数字: {ref_numbers_str})")
            
            # 解析所有数字
            ref_numbers = [int(n.strip()) for n in ref_numbers_str.split(',')]
            print(f"📝 引用数字: {ref_numbers}")
            
            # 创建对应的链接
            links = []
            for ref_num in ref_numbers:
                if ref_num <= len(all_sources) and ref_num > 0:
                    source = all_sources[ref_num - 1]  # 转换为0索引
                    title = source.get("title", "")
                    real_url = source.get("value", "")
                    
                    if title and title.strip():
                        # 清理标题，移除可能的域名后缀
                        clean_title = title.replace(".com", "").replace(".cn", "").replace(".net", "").replace(".org", "")
                        if not clean_title:
                            clean_title = title
                        link_text = clean_title
                    else:
                        link_text = f"来源{ref_num}"
                    
                    if real_url:
                        links.append(f"[{link_text}]({real_url})")
                        if source not in unique_sources:
                            unique_sources.append(source)
                else:
                    print(f"⚠️ 引用数字 {ref_num} 超出来源范围 (最大: {len(all_sources)})")
            
            if links:
                # 用链接替换引用
                replacement = " ".join(links) if len(links) > 1 else links[0]
                print(f"✅ 替换引用 {full_match} -> {replacement[:100]}...")
                
                # 使用match的位置信息进行精确替换
                start, end = match.span()
                result.content = result.content[:start] + replacement + result.content[end:]
        
        print(f"🔄 总共处理了 {len(matches)} 个引用，使用了 {len(unique_sources)} 个来源")
        
        # 策略2: 如果没有找到任何引用，尝试修复LLM可能创建的错误链接
        if len(unique_sources) == 0 and result.content:
            print("🔧 未找到标准引用格式，尝试修复可能的错误链接...")
            
            # 查找可能的markdown链接格式
            markdown_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', result.content)
            print(f"🔍 发现 {len(markdown_links)} 个markdown链接")
            
            for link_text, link_url in markdown_links:
                print(f"🔍 检查链接: [{link_text}]({link_url})")
                
                # 如果链接包含localhost或者是无效的，尝试替换
                if 'localhost' in link_url or link_url.startswith('http://localhost') or link_url == '#':
                    print(f"🚨 发现无效链接: {link_url}")
                    
                    # 尝试找到最合适的真实URL来替换
                    if all_sources:
                        # 使用第一个可用的真实URL
                        replacement_source = all_sources[0]
                        real_url = replacement_source.get("value", "")
                        if real_url:
                            print(f"🔄 替换无效链接为: {real_url[:50]}...")
                            # 保持原来的链接文本，只替换URL
                            result.content = result.content.replace(
                                f"[{link_text}]({link_url})", 
                                f"[{link_text}]({real_url})"
                            )
                            if replacement_source not in unique_sources:
                                unique_sources.append(replacement_source)
        
        # 策略3: 如果仍然没有引用，但有来源，则在答案末尾添加来源列表
        if len(unique_sources) == 0 and all_sources:
            print("📚 添加来源列表到答案末尾...")
            sources_section = "\n\n## 参考来源\n\n"
            for i, source in enumerate(all_sources[:5]):  # 最多显示5个来源
                real_url = source.get("value", "")
                title = source.get("title", "")
                
                if real_url:
                    # 创建更好的链接文本
                    if title and title.strip():
                        # 清理标题
                        clean_title = title.replace(".com", "").replace(".cn", "").replace(".net", "").replace(".org", "")
                        if not clean_title:
                            clean_title = title
                        link_text = clean_title
                    else:
                        link_text = f"来源{i+1}"
                    
                    sources_section += f"{i+1}. [{link_text}]({real_url})\n"
                    unique_sources.append(source)
            
            result.content += sources_section
            print(f"✅ 添加了 {len(unique_sources)} 个来源到答案末尾")

        print(f"✅ 生成的回复长度: {len(result.content) if result.content else 0}")
        print(f"🔗 处理的来源数量: {len(unique_sources)}")
        print(f"📝 替换后的回复前200字符:")
        print(f"   {result.content[:200] if result.content else 'None'}...")
        
        # 确保AIMessage有一个有效的ID
        ai_message = AIMessage(
            content=result.content if result.content else "抱歉，无法生成回复",
            id=f"ai-{int(__import__('time').time() * 1000)}"
        )
        print(f"📤 返回消息ID: {ai_message.id}")
        
        return {
            "messages": [ai_message],
            "sources_gathered": unique_sources,
        }
    except Exception as e:
        print(f"Error in finalize_answer: {e}")
        # 返回错误消息
        ai_message = AIMessage(
            content=f"抱歉，在生成最终答案时遇到错误: {str(e)}",
            id=f"ai-{int(__import__('time').time() * 1000)}"
        )
        return {
            "messages": [ai_message],
            "sources_gathered": [],
        }


def ensure_state_defaults(state: OverallState) -> OverallState:
    """确保状态包含所有必需的默认值"""
    defaults = {
        "search_query": [],
        "web_research_result": [],
        "sources_gathered": [],
        "research_loop_count": 0,
        "search_provider": "google",
        "llm_provider": "",  # 空字符串表示使用系统默认配置
        "is_sufficient": False,
        "follow_up_queries": [],
        "knowledge_gap": "",
        "number_of_ran_queries": 0,
        "messages": [],
    }
    
    for key, default_value in defaults.items():
        if key not in state:
            state[key] = default_value
    
    return state


# Create our Agent Graph
builder = StateGraph(OverallState, config_schema=Configuration)

# Define the nodes we will cycle between
builder.add_node("ensure_defaults", ensure_state_defaults)
builder.add_node("generate_query", generate_query)
builder.add_node("web_research", web_research)
builder.add_node("reflection", reflection)
builder.add_node("finalize_answer", finalize_answer)

# Set the entrypoint as `ensure_defaults`
# This means that this node is the first one called
builder.add_edge(START, "ensure_defaults")
builder.add_edge("ensure_defaults", "generate_query")
# Add conditional edge to continue with search queries in a parallel branch
builder.add_conditional_edges(
    "generate_query", continue_to_web_research, ["web_research"]
)
# Reflect on the web research
builder.add_edge("web_research", "reflection")
# Evaluate the research
builder.add_conditional_edges(
    "reflection", evaluate_research, ["web_research", "finalize_answer"]
)
# Finalize the answer
builder.add_edge("finalize_answer", END)

graph = builder.compile(name="pro-search-agent")
