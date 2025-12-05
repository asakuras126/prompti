#!/usr/bin/env python3
"""
OpenAI Client 使用示例

演示如何使用 OpenAI 兼容的模型客户端进行各种类型的调用。
"""

import asyncio
import os
import json

from prompti.model_client.base import ModelConfig, RunParams, ToolParams, ToolSpec
from prompti.model_client.openai_client import OpenAIClient
from prompti.message import Message, ModelResponse, StreamingResponse
from prompti.engine import PromptEngine, Setting

setting = Setting.from_file("./examples/configs/settings.yaml")
engine = PromptEngine.from_setting(setting)


async def basic_text_example():
    """基础文本对话示例"""
    print("=== 基础文本对话示例 ===")

    # 创建客户端
    client = OpenAIClient(cfg=engine.get_model_config("gpt-4o", "openai"))

    # 准备消息
    messages = [
        Message.create_system("你是一个有用的AI助手。"),
        Message.create_user_text("请简单介绍一下人工智能的发展历程。")
    ]

    # 非流式调用
    print("非流式响应:")
    params = RunParams(
        messages=messages,
        stream=False,
        max_tokens=200
    )

    async for response in client.run(params):
        if isinstance(response, ModelResponse):
            print(response.model_dump(exclude_none=True))

    print("\n" + "=" * 50 + "\n")

    # 流式调用
    print("流式响应:")
    stream_params = RunParams(
        messages=messages,
        stream=True,
        max_tokens=200
    )

    async for response in client.run(stream_params):
        if isinstance(response, StreamingResponse):
            print(response.model_dump(exclude_none=True))

    print("\n\n" + "=" * 50 + "\n")
    await client.close()


async def multimodal_example():
    """多模态（图像+文本）示例"""
    print("=== 多模态示例 ===")

    client = OpenAIClient(cfg=engine.get_model_config("gpt-4o", "openai"))

    messages = [
        {
            "role": "system",
            "content": "你是一个专业的图像分析师。"
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    }
                },
                {
                    "type": "text",
                    "text": "请描述这张图片中的内容"
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/2/25/Arbres_hivernaux_%281942%29_L%C3%A9on_Spilliaert_B000308.jpg",
                    }
                },
                {
                    "type": "text",
                    "text": "请描述这张图片中的内容"
                }
            ]
        }
    ]

    prompti_messages = Message.get_openai_messages(messages)

    params = RunParams(
        messages=prompti_messages,
        stream=False,
        max_tokens=300
    )

    async for response in client.run(params):
        if isinstance(response, ModelResponse):
            print(f"图像分析结果: {response.model_dump(exclude_none=True)}")

    print("\n" + "=" * 50 + "\n")

    # 流式调用
    print("流式响应:")
    stream_params = RunParams(
        messages=prompti_messages,
        stream=True,
        max_tokens=200
    )

    async for response in client.run(stream_params):
        if isinstance(response, StreamingResponse):
            print(response.model_dump(exclude_none=True))

    print("\n\n" + "=" * 50 + "\n")
    await client.close()


async def function_calling_example():
    """函数调用示例"""
    print("=== 函数调用示例 ===")

    client = OpenAIClient(cfg=engine.get_model_config("gpt-4o", "openai"))

    # 定义天气查询工具
    weather_tool = ToolSpec(
        name="get_weather",
        description="获取指定城市的当前天气信息",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "城市名称，例如：北京、上海、广州"
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "温度单位",
                    "default": "celsius"
                }
            },
            "required": ["city"]
        }
    )

    # 定义计算器工具
    calculator_tool = ToolSpec(
        name="calculate",
        description="执行基本的数学计算",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "要计算的数学表达式，例如：2+3*4"
                }
            },
            "required": ["expression"]
        }
    )

    tool_params = ToolParams(
        tools=[weather_tool, calculator_tool],
        choice="auto"  # 让AI自动选择是否使用工具
    )

    messages = [
        Message.create_system("你是一个智能助手，可以查询天气和进行数学计算。"),
        Message.create_user_text("请帮我查询一下北京的天气，然后计算 15 * 8 + 32 的结果。")
    ]

    params = RunParams(
        messages=messages,
        tool_params=tool_params,
        stream=False,
        max_tokens=500
    )

    print("发送请求...")

    async for response in client.run(params):
        if isinstance(response, ModelResponse):
            if response.error:
                print(f"错误: {response.error}")
                continue
            print(response)
            message = response.get_message()

            if message and message.has_tool_calls():
                print("AI 请求调用工具:")

                # 首先添加助手的工具调用消息
                messages.append(message)

                # 然后为每个工具调用创建对应的工具结果消息
                for tool_call in message.tool_calls:
                    function_name = tool_call.get("function", {}).get("name", "")
                    function_args = tool_call.get("function", {}).get("arguments", "{}")
                    tool_call_id = tool_call.get("id", "")

                    print(f"  工具: {function_name}")
                    print(f"  参数: {function_args}")
                    print(f"  ID: {tool_call_id}")

                    # 模拟工具执行
                    tool_result = await execute_tool(function_name, function_args)
                    print(f"  结果: {tool_result}")

                    # 创建工具结果消息
                    tool_result_message = Message.create_tool_result(
                        content=json.dumps(tool_result, ensure_ascii=False),
                        tool_call_id=tool_call_id
                    )
                    messages.append(tool_result_message)  # 添加工具结果消息

                # 如果有工具调用，再次请求获取最终回复
                if message.has_tool_calls():
                    print("\n获取最终回复...")
                    final_params = RunParams(
                        messages=messages,
                        tool_params=tool_params,
                        stream=False,
                        max_tokens=500
                    )

                    async for final_response in client.run(final_params):
                        if isinstance(final_response, ModelResponse):
                            if final_response.error:
                                print(f"错误: {final_response.error}")
                            else:
                                final_content = final_response.get_text_content()
                                print(f"最终回复: {final_content}")
            else:
                # 直接的文本回复
                content = response.get_text_content()
                print(f"回复: {content}")

    await client.close()


async def execute_tool(function_name: str, function_args: str) -> dict:
    """模拟工具执行"""
    try:
        args = json.loads(function_args)
    except json.JSONDecodeError:
        return {"error": "Invalid JSON arguments"}

    if function_name == "get_weather":
        city = args.get("city", "")
        unit = args.get("unit", "celsius")
        # 模拟天气查询结果
        return {
            "city": city,
            "temperature": 22 if unit == "celsius" else 72,
            "unit": unit,
            "condition": "晴天",
            "humidity": "65%"
        }
    elif function_name == "calculate":
        expression = args.get("expression", "")
        try:
            # 简单的数学计算（实际应用中应该使用更安全的方法）
            result = eval(expression)
            return {
                "expression": expression,
                "result": result
            }
        except Exception as e:
            return {"error": f"计算错误: {str(e)}"}
    else:
        return {"error": f"Unknown function: {function_name}"}


async def streaming_function_calling_example():
    """流式函数调用示例"""
    print("\n=== 流式函数调用示例 ===")

    client = OpenAIClient(cfg=engine.get_model_config("gpt-4o", "openai"))

    # 定义一个简单的工具
    time_tool = ToolSpec(
        name="get_current_time",
        description="获取当前时间",
        parameters={
            "type": "object",
            "properties": {
                "timezone": {
                    "type": "string",
                    "description": "时区，例如：Asia/Shanghai",
                    "default": "Asia/Shanghai"
                }
            }
        }
    )

    tool_params = ToolParams(
        tools=[time_tool],
        choice="auto"
    )

    messages = [
        Message.create_system("你是一个时间助手。"),
        Message.create_user_text("现在几点了？")
    ]

    params = RunParams(
        messages=messages,
        tool_params=tool_params,
        stream=True
    )

    print("流式处理中...")

    tool_calls_buffer = {}

    async for response in client.run(params):
        if isinstance(response, StreamingResponse):
            if response.error:
                print(f"错误: {response.error}")
                continue

            delta = response.get_delta()
            if delta and delta.tool_calls:
                # 处理流式工具调用
                for tool_call in delta.tool_calls:
                    call_id = tool_call.get("id", "")
                    if call_id not in tool_calls_buffer:
                        tool_calls_buffer[call_id] = {
                            "id": call_id,
                            "type": tool_call.get("type", "function"),
                            "function": {
                                "name": "",
                                "arguments": ""
                            }
                        }

                    function_data = tool_call.get("function", {})
                    if "name" in function_data:
                        tool_calls_buffer[call_id]["function"]["name"] += function_data["name"]
                    if "arguments" in function_data:
                        tool_calls_buffer[call_id]["function"]["arguments"] += function_data["arguments"]

                print(f"工具调用进度: {tool_calls_buffer}")

            elif delta and delta.content:
                print(f"内容片段: {delta.content}")

    # 处理完整的工具调用
    if tool_calls_buffer:
        print("\n完整的工具调用:")
        for call_id, tool_call in tool_calls_buffer.items():
            print(f"  {tool_call}")

    await client.close()


async def advanced_parameters_example():
    """高级参数示例"""
    print("=== 高级参数示例 ===")

    client = OpenAIClient(cfg=engine.get_model_config("gpt-4o", "openai"))

    messages = [
        Message.create_system("你是一个创意写作助手。"),
        Message.create_user_text("写一首关于春天的短诗。")
    ]

    # 使用高级参数
    params = RunParams(
        messages=messages,
        stream=True,
        temperature=0.9,  # 更高的创造性
        top_p=0.9,
        max_tokens=150,
        stop=["。"],  # 遇到。停止
        seed=42,  # 可重现的结果
        extra_params={
            "frequency_penalty": 0.5,  # 减少重复
            "presence_penalty": 0.3  # 鼓励多样性
        }
    )

    print("创意写作结果:")
    async for response in client.run(params):
        if isinstance(response, StreamingResponse):
            content = response.get_text_content()
            if content:
                print(content, end="", flush=True)

    print("\n\n" + "=" * 50 + "\n")
    await client.close()


async def main():
    """主函数"""
    print("OpenAI Client 使用示例\n")

    try:
        # 运行各种示例
        await basic_text_example()

        # 多模态示例（需要支持视觉的模型）
        await multimodal_example()

        await function_calling_example()
        # await streaming_function_calling_example()

        # await advanced_parameters_example()

    except Exception as e:
        print(f"示例运行出错: {e}")


if __name__ == "__main__":
    asyncio.run(main())
