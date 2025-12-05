"""
Sticky Routing Helper - Session级别粘性路由辅助模块

提供内联式的sticky routing功能，不使用wrapper模式。
主要用于Engine在路由选择前后进行Redis的读写操作。

核心功能:
1. Redis客户端管理 (支持配置合并)
2. 获取已存储的sticky variant
3. 存储新选择的variant

Redis Key格式:
    {key_prefix}:{template_name}:{session_id}

使用场景:
    - Engine在执行路由前，先查询Redis是否有sticky variant
    - 如果有，直接返回，不执行路由
    - 如果没有，执行路由后将结果存入Redis
"""

from __future__ import annotations
from typing import Any, Optional

from ..logger import get_logger

logger = get_logger(__name__)


class StickyRoutingHelper:
    """Sticky routing辅助类 - 处理Redis读写和配置管理"""

    @staticmethod
    def _build_final_config(sticky_config: dict[str, Any], global_config_getter=None) -> dict[str, Any]:
        """构建最终的 sticky routing 配置

        合并全局配置和 sticky_config,设置默认值

        Args:
            sticky_config: Sticky routing配置
            global_config_getter: 全局配置获取函数 (可选)

        Returns:
            合并后的配置字典
        """
        final_config = {}

        # 1. 从全局配置获取Redis基础配置
        if global_config_getter:
            global_redis_config = global_config_getter("redis")
            if global_redis_config and global_redis_config.get("enabled"):
                final_config.update({
                    "redis_host": global_redis_config.get("host", "localhost"),
                    "redis_port": global_redis_config.get("port", 6379),
                    "redis_db": global_redis_config.get("db", 0),
                    "redis_password": global_redis_config.get("password"),
                })

            # 2. 从全局配置获取sticky_routing配置
            global_sticky_config = global_config_getter("sticky_routing")
            if global_sticky_config and global_sticky_config.get("enabled"):
                final_config.update({
                    "ttl_seconds": global_sticky_config.get("ttl_days", 30) * 24 * 3600,
                    "key_prefix": global_sticky_config.get("key_prefix", "sticky_routing"),
                    "template_field": global_sticky_config.get("template_field", "template_name"),
                    "auto_refresh_ttl": global_sticky_config.get("auto_refresh_ttl", True),
                })

        # 3. 用sticky_config覆盖 (最高优先级)
        if sticky_config:
            # Redis connection config
            if "redis_host" in sticky_config:
                final_config["redis_host"] = sticky_config["redis_host"]
            if "redis_port" in sticky_config:
                final_config["redis_port"] = sticky_config["redis_port"]
            if "redis_db" in sticky_config:
                final_config["redis_db"] = sticky_config["redis_db"]
            if "redis_password" in sticky_config:
                final_config["redis_password"] = sticky_config["redis_password"]

            # Sticky routing config
            if "ttl_days" in sticky_config:
                final_config["ttl_seconds"] = sticky_config["ttl_days"] * 24 * 3600
            if "key_prefix" in sticky_config:
                final_config["key_prefix"] = sticky_config["key_prefix"]
            if "template_field" in sticky_config:
                final_config["template_field"] = sticky_config["template_field"]
            if "auto_refresh_ttl" in sticky_config:
                final_config["auto_refresh_ttl"] = sticky_config["auto_refresh_ttl"]

        # 4. 设置默认值
        final_config.setdefault("redis_host", "localhost")
        final_config.setdefault("redis_port", 6379)
        final_config.setdefault("redis_db", 0)
        final_config.setdefault("ttl_seconds", 7 * 24 * 3600)
        final_config.setdefault("key_prefix", "sticky_routing")
        final_config.setdefault("template_field", "template_name")
        final_config.setdefault("auto_refresh_ttl", True)

        return final_config

    @staticmethod
    def get_redis_client(sticky_config: dict[str, Any], global_config_getter=None):
        """获取或创建Redis客户端

        Args:
            sticky_config: Sticky routing配置
            global_config_getter: 全局配置获取函数 (可选)

        Returns:
            (redis_client, final_config) 元组，或 None 如果连接失败
        """
        try:
            import redis
        except ImportError:
            logger.warning("Redis not installed, sticky routing disabled")
            return None

        # 构建最终配置 (复用公共方法)
        final_config = StickyRoutingHelper._build_final_config(sticky_config, global_config_getter)

        # 检查是否有有效的Redis host
        if not final_config.get("redis_host"):
            logger.warning("No Redis host configured, sticky routing disabled")
            return None

        # 创建Redis客户端
        try:
            redis_client = redis.Redis(
                host=final_config["redis_host"],
                port=final_config["redis_port"],
                db=final_config["redis_db"],
                password=final_config.get("redis_password"),
                decode_responses=True,
                socket_connect_timeout=1,
                socket_timeout=1,
            )
            # 测试连接
            redis_client.ping()
            logger.debug(
                f"Redis connected for sticky routing: "
                f"{final_config['redis_host']}:{final_config['redis_port']}"
            )
            return redis_client, final_config
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}, sticky routing disabled")
            return None

    @staticmethod
    def build_redis_key(
        session_id: str,
        template_name: str,
        key_prefix: str = "sticky_routing",
        route_name: str | None = None
    ) -> str:
        """构建Redis key

        Args:
            session_id: Session ID
            template_name: Template Name
            key_prefix: Key前缀
            route_name: Route名称 (可选，用于route-level sticky)

        Returns:
            Redis key字符串

        Format:
            - Route-level: {key_prefix}:{template_name}:{route_name}:{session_id}
            - Legacy: {key_prefix}:{template_name}:{session_id}
        """
        if route_name:
            return f"{key_prefix}:{template_name}:{route_name}:{session_id}"
        return f"{key_prefix}:{template_name}:{session_id}"

    @staticmethod
    def get_sticky_variant(
        sticky_config: dict[str, Any],
        route_context: dict[str, Any],
        template_name: str,
        route_name: str | None = None,
        global_config_getter=None,
        redis_client=None
    ) -> Optional[str]:
        """从Redis获取已存储的sticky variant

        Args:
            sticky_config: Sticky routing配置
            route_context: 路由上下文 (必须包含session_id)
            template_name: Template Name
            route_name: Route Name (可选，用于route-level sticky)
            global_config_getter: 全局配置获取函数
            redis_client: Redis客户端实例 (可选,如果提供则复用,否则创建新的)

        Returns:
            Variant名称，如果未找到返回None
        """
        # 检查session_id是否存在
        session_id = route_context.get("session_id")
        if not session_id:
            logger.debug("session_id not found in route_context, skipping sticky check")
            return None

        # 获取Redis客户端 (优先使用传入的client)
        if redis_client is None:
            redis_result = StickyRoutingHelper.get_redis_client(sticky_config, global_config_getter)
            if not redis_result:
                return None
            redis_client, final_config = redis_result
        else:
            # 使用传入的client,仍需要构建final_config
            final_config = StickyRoutingHelper._build_final_config(sticky_config, global_config_getter)

        # 确定template_name (优先从context获取)
        template_field = final_config.get("template_field", "template_name")
        resolved_template_name = route_context.get(template_field) or template_name

        # 生成Redis key
        redis_key = StickyRoutingHelper.build_redis_key(
            session_id,
            resolved_template_name,
            final_config.get("key_prefix", "sticky_routing"),
            route_name
        )

        try:
            variant = redis_client.get(redis_key)
            if variant:
                # 自动续期TTL
                if final_config.get("auto_refresh_ttl", True):
                    ttl_seconds = final_config.get("ttl_seconds", 7 * 24 * 3600)
                    redis_client.expire(redis_key, ttl_seconds)
                logger.debug(f"Sticky hit: {redis_key} -> {variant}")
                return variant
        except Exception as e:
            logger.warning(f"Failed to get sticky variant from Redis: {e}")

        return None

    @staticmethod
    def store_sticky_variant(
        sticky_config: dict[str, Any],
        route_context: dict[str, Any],
        template_name: str,
        variant_name: str,
        route_name: str | None = None,
        global_config_getter=None,
        redis_client=None
    ) -> bool:
        """存储variant到Redis

        Args:
            sticky_config: Sticky routing配置
            route_context: 路由上下文 (必须包含session_id)
            template_name: Template Name
            variant_name: 要存储的variant名称
            route_name: Route Name (可选，用于route-level sticky)
            global_config_getter: 全局配置获取函数
            redis_client: Redis客户端实例 (可选,如果提供则复用,否则创建新的)

        Returns:
            True如果存储成功，False如果失败
        """
        # 检查session_id是否存在
        session_id = route_context.get("session_id")
        if not session_id:
            logger.debug("session_id not found in route_context, skipping sticky store")
            return False

        # 获取Redis客户端 (优先使用传入的client)
        if redis_client is None:
            redis_result = StickyRoutingHelper.get_redis_client(sticky_config, global_config_getter)
            if not redis_result:
                return False
            redis_client, final_config = redis_result
        else:
            # 使用传入的client,仍需要构建final_config
            final_config = StickyRoutingHelper._build_final_config(sticky_config, global_config_getter)

        # 确定template_name (优先从context获取)
        template_field = final_config.get("template_field", "template_name")
        resolved_template_name = route_context.get(template_field) or template_name

        # 生成Redis key
        redis_key = StickyRoutingHelper.build_redis_key(
            session_id,
            resolved_template_name,
            final_config.get("key_prefix", "sticky_routing"),
            route_name
        )

        try:
            ttl_seconds = final_config.get("ttl_seconds", 7 * 24 * 3600)
            redis_client.setex(redis_key, ttl_seconds, variant_name)
            logger.debug(f"Sticky stored: {redis_key} -> {variant_name} (TTL: {ttl_seconds}s)")
            return True
        except Exception as e:
            logger.warning(f"Failed to store sticky variant to Redis: {e}")
            return False

    @staticmethod
    def clear_sticky_variant(
        sticky_config: dict[str, Any],
        session_id: str,
        template_name: str,
        route_name: str | None = None,
        global_config_getter=None
    ) -> bool:
        """清除指定的sticky variant记录

        Args:
            sticky_config: Sticky routing配置
            session_id: Session ID
            template_name: Template Name
            route_name: Route Name (可选，用于route-level sticky)
            global_config_getter: 全局配置获取函数

        Returns:
            True如果删除成功，False如果失败
        """
        redis_result = StickyRoutingHelper.get_redis_client(sticky_config, global_config_getter)
        if not redis_result:
            return False

        redis_client, final_config = redis_result

        redis_key = StickyRoutingHelper.build_redis_key(
            session_id,
            template_name,
            final_config.get("key_prefix", "sticky_routing"),
            route_name
        )

        try:
            deleted = redis_client.delete(redis_key)
            logger.debug(f"Cleared sticky: {redis_key} (deleted: {deleted})")
            return deleted > 0
        except Exception as e:
            logger.warning(f"Failed to clear sticky variant: {e}")
            return False
