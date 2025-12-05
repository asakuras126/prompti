#!/usr/bin/env python3
"""同步工具调用脱敏反脱敏示例。"""


from prompti.engine import PromptEngine, Setting, Message

from prompti.model_client.base import ModelConfig, RunParams, ToolParams, ToolSpec
import logging
try:
    from loguru import logger


    class InterceptHandler(logging.Handler):
        def emit(self, record):
            # 将 logging 的消息转给 loguru
            logger_opt = logger.opt(depth=6, exception=record.exc_info)
            logger_opt.log(record.levelname, record.getMessage())


    logging.basicConfig(handlers=[InterceptHandler()], level=logging.INFO)
except ImportError:
    logging.basicConfig(level=logging.INFO)

def main():
    """主函数。"""
    # 配置敏感词表
    wordlist = {
        "张三": "User001",
        "李四": "User002",
        "王五": "User003",
        "13812345678": "Phone001",
        "15987654321": "Phone002",
        "zhangsan@company.com": "Email001",
        "lisi@company.com": "Email002",
        "北京市朝阳区建国门外大街1号": "Address001",
        "XX科技有限公司": "Company001"
    }
    

    # 定义工具
    tools = [
        ToolSpec(

                name="send_email",
                description="发送邮件",
                parameters={
                    "type": "object",
                    "properties": {
                        "recipient": {"type": "string", "description": "收件人邮箱"},
                        "sender": {"type": "string", "description": "发件人"},
                        "subject": {"type": "string", "description": "邮件主题"},
                        "content": {"type": "string", "description": "邮件内容"}
                    },
                    "required": ["recipient", "content"]
                }

        ),
        ToolSpec(

                name="update_contact",
                description="更新联系人信息",
                parameters={
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "姓名"},
                        "phone": {"type": "string", "description": "电话"},
                        "email": {"type": "string", "description": "邮箱"},
                        "company": {"type": "string", "description": "公司"}
                    },
                    "required": ["name"]

            }
        )
    ]
    


    setting = Setting(
        registry_url="http://10.224.55.241/api/v1",
        registry_api_key="7e5d106c-e701-4587-a95a-b7c7c02ee619",
    )
    # 创建引擎
    engine = PromptEngine.from_setting(setting)
    
    # 测试请求
    request = "请给zhangsan@company.com发送一封邮件，发件人是李四，主题是会议通知，内容是请张三明天下午2点到北京市朝阳区建国门外大街1号参加会议，如有问题请联系13812345678。"
    
    try:
        # 执行调用
        for response in engine.completion(
            template_name="simple-demo",
            variables={"instruction": "",
                           "query": request, "chat_history": "",
                    "chat_log":"123",
                           },
            tool_params=tools,
            variant_name="default",
            hook_configs={
                "desensitization": {
                    "type": "wordlist",
                    "wordlist": wordlist
                }

            },
                stream=True
        ):

            logger.info(f"user get response: {response}")
    except Exception as e:
        logger.exception(f"错误: {e}")
    

if __name__ == "__main__":
    main()