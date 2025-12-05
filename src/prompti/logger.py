"""
Prompti 统一日志模块

提供统一的日志接口,支持:
1. 用户传入自定义 logger
2. 默认使用 miaoda_logger (如果可用,优先使用)
3. fallback 到 loguru (输出到控制台)
4. 通过环境变量 PROMPTI_LOG_LEVEL 配置日志级别
5. 最终 fallback 到标准 logging

环境变量:
    PROMPTI_LOG_LEVEL: 日志级别 (DEBUG/INFO/WARNING/ERROR),默认 INFO

示例:
    # 默认使用
    from prompti.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Hello")

    # 设置自定义 logger
    from prompti.logger import set_logger
    from loguru import logger as my_logger
    set_logger(my_logger)
"""

import logging
import os
import sys
from typing import Any, Optional, Union

# 日志接口协议 (支持标准logging/loguru/miaoda_logger)
LoggerProtocol = Union[logging.Logger, Any]

# 全局用户自定义logger
_user_logger: Optional[LoggerProtocol] = None

# 默认logger实例缓存
_default_logger: Optional[LoggerProtocol] = None

# 支持的日志级别
LOG_LEVELS = {
    "DEBUG": "DEBUG",
    "INFO": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "CRITICAL": "CRITICAL",
}


def _get_log_level_from_env() -> str:
    """
    从环境变量获取日志级别

    环境变量: PROMPTI_LOG_LEVEL
    支持值: DEBUG, INFO, WARNING, ERROR, CRITICAL
    默认值: INFO

    Returns:
        日志级别字符串 (大写)
    """
    level = os.getenv("PROMPTI_LOG_LEVEL", "INFO").upper()
    if level not in LOG_LEVELS:
        print(f"Warning: Invalid PROMPTI_LOG_LEVEL '{level}', using INFO", file=sys.stderr)
        return "INFO"
    return level


def set_logger(logger: LoggerProtocol) -> None:
    """
    设置全局自定义 logger

    Args:
        logger: 用户自定义的 logger 实例,需要支持标准的 logging 接口
                (debug, info, warning, error, exception 方法)

    Example:
        >>> from loguru import logger as my_logger
        >>> my_logger.add("app.log", rotation="500 MB")
        >>> set_logger(my_logger)

        >>> import logging
        >>> my_logger = logging.getLogger("my_app")
        >>> my_logger.setLevel(logging.INFO)
        >>> set_logger(my_logger)
    """
    global _user_logger
    _user_logger = logger


def get_logger(name: Optional[str] = None) -> LoggerProtocol:
    """
    获取 logger 实例

    优先级:
    1. 用户通过 set_logger() 设置的自定义 logger
    2. miaoda_logger (如果可用,优先使用)
    3. loguru (默认推荐,输出到控制台)
    4. 标准 logging (最终 fallback)

    日志级别通过环境变量 PROMPTI_LOG_LEVEL 控制,默认 INFO

    Args:
        name: logger 名称,用于标准 logging,对 loguru 无效

    Returns:
        logger 实例
    """
    global _default_logger, _user_logger

    # 如果用户设置了自定义logger,直接使用
    if _user_logger is not None:
        return _user_logger

    # 如果已经初始化过默认logger,直接返回
    if _default_logger is not None:
        return _default_logger

    # 初始化默认logger (只执行一次)
    _default_logger = _init_default_logger(name)
    return _default_logger


def _init_default_logger(name: Optional[str] = None) -> LoggerProtocol:
    """
    初始化默认 logger

    优先级: miaoda_logger > loguru > logging
    """
    log_level = _get_log_level_from_env()

    # 2. 尝试使用 loguru (推荐)
    try:
        from loguru import logger as loguru_logger

        # 移除默认handler
        loguru_logger.remove()

        # 只添加控制台输出 (stderr)
        loguru_logger.add(
            sys.stderr,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                   "<level>{level: <8}</level> | "
                   "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                   "<level>{message}</level>",
            level=log_level,
            colorize=True,
        )
        return loguru_logger
    except ImportError:
        pass

    # 3. 最终 fallback 到标准 logging
    logger = logging.getLogger(name or "prompti")
    if not logger.handlers:
        # 只添加控制台输出 (stderr)
        handler = logging.StreamHandler(sys.stderr)
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # 设置日志级别
        numeric_level = getattr(logging, log_level, logging.INFO)
        logger.setLevel(numeric_level)

    return logger


def reset_logger() -> None:
    """
    重置 logger 配置 (主要用于测试)
    """
    global _user_logger, _default_logger
    _user_logger = None
    _default_logger = None


# 便捷函数:直接导出常用方法 (可选)
def debug(msg: str, *args, **kwargs):
    """便捷函数:输出 debug 日志"""
    get_logger().debug(msg, *args, **kwargs)


def info(msg: str, *args, **kwargs):
    """便捷函数:输出 info 日志"""
    get_logger().info(msg, *args, **kwargs)


def warning(msg: str, *args, **kwargs):
    """便捷函数:输出 warning 日志"""
    get_logger().warning(msg, *args, **kwargs)


def error(msg: str, *args, **kwargs):
    """便捷函数:输出 error 日志"""
    get_logger().error(msg, *args, **kwargs)


def exception(msg: str, *args, **kwargs):
    """便捷函数:输出 exception 日志"""
    get_logger().exception(msg, *args, **kwargs)
