"""
工具追踪使用示例 - 主要演示 engine.report_tool_trace() 方法
"""

import time
import random
import sys
import os

# 添加 prompti src 到 Python 路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from prompti.engine import PromptEngine, Setting


def main():
    """主要示例：使用 PromptEngine 的 report_tool_trace 方法"""
    print("🚀 开始工具追踪示例...")
    
    # 1. 创建 Setting 配置（工具追踪复用registry_url）
    setting = Setting(
        registry_url="http://10.224.55.241/api/v1",  # LLM追踪和工具追踪共用
        registry_api_key="7e5d106c-e701-4587-a95a-b7c7c02ee619",
    )
    
    # 2. 从 Setting 创建 PromptEngine（自动包含工具追踪服务）
    engine = PromptEngine.from_setting(setting)
    print("✅ PromptEngine 创建成功，工具追踪服务已配置")
    
    # 3. 生成测试数据
    conversation_id = "conv-" + "".join(random.choices("0123456789abcdef", k=12))
    user_id = "user-" + "".join(random.choices("0123456789abcdef", k=8))
    app_id = "tool-trace-demo"
    
    print(f"📝 测试会话ID: {conversation_id}")
    print(f"👤 测试用户ID: {user_id}")
    print(f"🔧 追踪服务URL: {setting.registry_url}")
    print(f"🔧 工具追踪服务状态: {'已配置' if engine._tool_trace_service else '未配置'}")
    
    # 4. 示例1：成功的工具调用
    print("\n1️⃣ 测试成功的工具调用...")
    success_payload = {
        "name": "web_search",
        "inputs": {
            "query": "Python asyncio tutorial",
            "max_results": 5,
            "language": "zh-CN"
        },
        "outputs": {
            "results": [
                {"title": "Python Asyncio 完全指南", "url": "https://example1.com", "score": 0.95},
                {"title": "异步编程最佳实践", "url": "https://example2.com", "score": 0.87},
                {"title": "Asyncio 性能优化技巧", "url": "https://example3.com", "score": 0.82}
            ],
            "total_count": 3,
            "search_time_ms": 127
        },
        "start_at": time.time() - 0.15,
        "end_at": time.time() - 0.02,
        "request_id": f"req-{random.randint(1000, 9999)}",
        "conversation_id": conversation_id,
        "user_id": user_id,
        "app_id": app_id,
        "source": "prompti-example"
    }
    
    success = engine.report_tool_trace(success_payload)
    print(f"   ✅ 成功工具调用上报: {'成功' if success else '失败'}")
    
    time.sleep(0.2)
    
    # 5. 示例2：失败的工具调用
    print("\n2️⃣ 测试失败的工具调用...")
    error_payload = {
        "name": "database_query",
        "inputs": {
            "sql": "SELECT * FROM user_profiles WHERE active = 1",
            "database": "production",
            "timeout": 30
        },
        "error": "DatabaseConnectionError: Connection timeout after 30 seconds. Unable to connect to production database server.",
        "start_at": time.time() - 30.5,
        "end_at": time.time() - 0.1,
        "request_id": f"req-{random.randint(1000, 9999)}",
        "conversation_id": conversation_id,
        "user_id": user_id,
        "app_id": app_id,
        "source": "prompti-example"
    }
    
    success = engine.report_tool_trace(error_payload)
    print(f"   ❌ 失败工具调用上报: {'成功' if success else '失败'}")
    
    time.sleep(0.2)
    
    # 6. 示例3：带层级关系的工具调用
    print("\n3️⃣ 测试带层级关系的工具调用...")
    parent_span = f"span-parent-{random.randint(100, 999)}"
    child_span = f"span-child-{random.randint(100, 999)}"
    
    # 父级工具调用
    parent_payload = {
        "name": "user_authentication",
        "inputs": {
            "username": "john_doe", 
            "auth_method": "oauth2"
        },
        "outputs": {
            "user_id": "user_12345",
            "access_token": "tok_***redacted***",
            "expires_in": 3600
        },
        "start_at": time.time() - 1.2,
        "end_at": time.time() - 1.0,
        "request_id": f"req-{random.randint(1000, 9999)}",
        "conversation_id": conversation_id,
        "user_id": user_id,
        "app_id": app_id,
        "span_id": parent_span,
        "source": "prompti-example"
    }
    
    success = engine.report_tool_trace(parent_payload)
    print(f"   👨‍💼 父级工具调用上报: {'成功' if success else '失败'}")
    
    # 子级工具调用
    child_payload = {
        "name": "fetch_user_preferences",
        "inputs": {
            "user_id": "user_12345",
            "include_settings": True
        },
        "outputs": {
            "preferences": {
                "theme": "dark",
                "language": "zh-CN", 
                "notifications": True
            },
            "last_updated": "2025-01-20T10:30:00Z"
        },
        "start_at": time.time() - 0.8,
        "end_at": time.time() - 0.3,
        "request_id": f"req-{random.randint(1000, 9999)}",
        "conversation_id": conversation_id,
        "user_id": user_id,
        "app_id": app_id,
        "span_id": child_span,
        "parent_span_id": parent_span,  # 指向父级
        "source": "prompti-example"
    }
    
    success = engine.report_tool_trace(child_payload)
    print(f"   👶 子级工具调用上报: {'成功' if success else '失败'}")
    
    time.sleep(0.2)
    
    # 7. 示例4：最小化的工具调用（只有必填字段）
    print("\n4️⃣ 测试最小化工具调用...")
    minimal_payload = {
        "name": "simple_calculation", 
        "inputs": {"expression": "2 + 2"},
        "outputs": {"result": 4},
        "conversation_id": conversation_id,
        "user_id": user_id,
        "app_id": app_id
    }
    
    success = engine.report_tool_trace(minimal_payload)
    print(f"   ⚡ 最小化工具调用上报: {'成功' if success else '失败'}")
    
    # 8. 等待异步上报完成
    print(f"\n⏳ 等待异步上报完成...")
    time.sleep(2)
    
    print(f"\n🎉 所有工具追踪示例执行完成！")
    print(f"📊 可以在 promptstore 中查看会话 {conversation_id} 的工具调用记录")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"❌ 示例执行出错: {e}")
        import traceback
        traceback.print_exc()