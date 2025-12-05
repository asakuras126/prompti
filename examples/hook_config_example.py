#!/usr/bin/env python3
"""演示如何使用配置文件设置Hook的示例"""

from pathlib import Path
import yaml
from prompti.engine import PromptEngine, Setting
from prompti.model_client import ModelConfig

def create_config_file():
    """创建示例配置文件"""
    
    config = {
        "cache_ttl": 300,
        
        # 安全分类Hook配置
        "safety_hook_config": {
            "strategy": "check_non_domestic",
            "api_url": "https://aisecurity.baidu-int.com/output_safety_multi_classification_service",
            "timeout": 5,
            "max_retries": 3,
            "retry_delay": 1.0,
            "max_concurrent_checks": 10,
            "blocked_message": "内容包含不当信息，已被安全过滤"
        },
        
        # 脱敏Hook配置
        "anonymization_hook_config": {
            "type": "wordlist",
            "wordlist": {
                # 个人信息脱敏
                "张三": "[用户A]",
                "李四": "[用户B]", 
                "13800138000": "[手机号]",
                "18600186000": "[手机号]",
                
                # 敏感词脱敏
                "机密文件": "[文档]",
                "内部资料": "[资料]",
                "秘密项目": "[项目]",
                
                # 公司信息脱敏  
                "百度公司": "[公司A]",
                "阿里巴巴": "[公司B]",
                "腾讯科技": "[公司C]"
            }
        },
        
        # 自定义Hook配置（可用于扩展）
        "default_hook_configs": {
            "custom_logging": {
                "enabled": True,
                "log_level": "INFO",
                "log_requests": True,
                "log_responses": False
            }
        }
    }
    
    config_file = Path("hook_settings.yaml")
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, indent=2)
    
    print(f"✓ 创建配置文件: {config_file}")
    return config_file

def demo_hook_configuration():
    """演示Hook配置的使用"""
    
    # 1. 创建配置文件
    config_file = create_config_file()
    
    try:
        # 2. 从配置文件加载Setting
        setting = Setting.from_file(str(config_file))
        print("✓ 加载配置文件成功")
        
        # 3. 查看配置内容
        print("\n配置内容:")
        print(f"  安全Hook策略: {setting.safety_hook_config.get('strategy')}")
        print(f"  脱敏词汇数量: {len(setting.anonymization_hook_config.get('wordlist', {}))}")
        print(f"  自定义配置: {setting.default_hook_configs}")
        
        # 4. 创建引擎（注意：这里会跳过网络请求部分）
        print("\n创建PromptEngine...")
        
        # 使用内存loader避免网络请求
        from prompti.loader import MemoryLoader
        
        engine = PromptEngine(
            prompt_loaders=[MemoryLoader({})],
            model_loaders=[],
            cache_ttl=setting.cache_ttl,
            default_hook_configs={
                "safety_classification": setting.safety_hook_config,
                "anonymization": setting.anonymization_hook_config,
                **setting.default_hook_configs
            }
        )
        
        print("✓ PromptEngine创建成功")
        
        # 5. 测试Hook配置合并
        print("\n测试Hook配置:")
        
        # 场景1: 使用默认配置
        print("  场景1: 使用配置文件中的默认Hook设置")
        before_hooks, after_hooks = engine._create_hooks_from_configs(None)
        print(f"    创建的Hook数量 - Before: {len(before_hooks)}, After: {len(after_hooks)}")
        
        # 场景2: 运行时覆盖部分配置
        print("  场景2: 运行时覆盖安全Hook策略")
        runtime_config = {
            "safety_classification": {
                "strategy": "all",  # 覆盖默认的check_non_domestic策略
                "timeout": 8        # 覆盖默认的5秒超时
            }
        }
        before_hooks2, after_hooks2 = engine._create_hooks_from_configs(runtime_config)
        print(f"    创建的Hook数量 - Before: {len(before_hooks2)}, After: {len(after_hooks2)}")
        
        # 场景3: 添加新的Hook配置
        print("  场景3: 添加新的Hook类型")
        extended_config = {
            "new_custom_hook": {
                "type": "example",
                "param": "value"
            }
        }
        before_hooks3, after_hooks3 = engine._create_hooks_from_configs(extended_config)
        print(f"    创建的Hook数量 - Before: {len(before_hooks3)}, After: {len(after_hooks3)}")
        
        # 6. 演示Hook类型
        print("\n已创建的Hook类型:")
        all_hooks = before_hooks + after_hooks
        for i, hook in enumerate(all_hooks):
            hook_type = type(hook).__name__
            print(f"  {i+1}. {hook_type}")
            if hasattr(hook, 'wordlist') and hook.wordlist:
                print(f"     - 脱敏词汇: {list(hook.wordlist.keys())[:3]}...")
            if hasattr(hook, 'api_url'):
                print(f"     - API地址: {hook.api_url}")
        
        print("\n✓ Hook配置演示完成!")
        return True
        
    except Exception as e:
        print(f"✗ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # 清理配置文件
        if config_file.exists():
            config_file.unlink()
            print(f"✓ 清理配置文件: {config_file}")

def demo_programmatic_configuration():
    """演示通过代码配置Hook"""
    
    print("\n" + "=" * 50)
    print("通过代码配置Hook示例:")
    
    try:
        # 直接在代码中配置
        setting = Setting(
            cache_ttl=300,
            
            # 启用所有模型的安全检查
            safety_hook_config={
                "strategy": "all",
                "timeout": 3,
                "max_retries": 2
            },
            
            # 配置脱敏规则
            anonymization_hook_config={
                "type": "wordlist",
                "wordlist": {
                    "开发者": "[开发人员]",
                    "项目经理": "[管理人员]",
                    "API密钥": "[密钥]"
                }
            }
        )
        
        print("✓ 通过代码创建Setting对象")
        print(f"  安全Hook策略: {setting.safety_hook_config['strategy']}")
        print(f"  脱敏配置: {len(setting.anonymization_hook_config['wordlist'])} 个词汇")
        
        return True
        
    except Exception as e:
        print(f"✗ 代码配置失败: {e}")
        return False

def main():
    """主函数"""
    print("PromptI Hook配置示例")
    print("=" * 50)
    
    # 演示1: 配置文件方式
    success1 = demo_hook_configuration()
    
    # 演示2: 代码配置方式  
    success2 = demo_programmatic_configuration()
    
    print("\n" + "=" * 50)
    if success1 and success2:
        print("✓ 所有演示完成!")
        print("\n使用说明:")
        print("1. 在配置文件中设置 safety_hook_config 和 anonymization_hook_config")
        print("2. 使用 Setting.from_file() 加载配置")
        print("3. 使用 PromptEngine.from_setting() 创建引擎")
        print("4. 运行时可通过 completion() 的 hook_configs 参数覆盖默认配置")
    else:
        print("✗ 部分演示失败!")

if __name__ == "__main__":
    main()