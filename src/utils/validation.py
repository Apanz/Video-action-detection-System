"""
输入验证工具模块
提供装饰器用于验证张量形状、设备、参数范围等
"""

import functools
import torch
import os
from typing import List, Tuple, Optional, Callable, Any


# 环境变量控制验证开关
VALIDATION_ENABLED = os.getenv('VALIDATION_ENABLED', 'false').lower() == 'true'


def validate_tensor_shape(expected_dims: List[int], arg_positions: Optional[List[int]] = None,
                          min_dim_size: Optional[int] = None):
    """
    装饰器：验证张量维度

    Args:
        expected_dims: 期望的维度列表，例如[4, 5]表示接受4维或5维张量
        arg_positions: 要验证的参数位置列表（默认验证第一个张量参数）
        min_dim_size: 每个维度的最小尺寸要求（可选）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 如果验证未启用，直接执行函数
            if not VALIDATION_ENABLED:
                return func(*args, **kwargs)

            # 确定要验证的参数位置
            positions = arg_positions if arg_positions is not None else [0]

            for pos in positions:
                # 获取参数（处理args和kwargs）
                if pos < len(args):
                    arg = args[pos]
                else:
                    continue

                # 检查是否为张量
                if isinstance(arg, torch.Tensor):
                    actual_dims = arg.dim()

                    # 验证维度
                    if actual_dims not in expected_dims:
                        raise ValueError(
                            f"[Validation] {func.__name__}(): "
                            f"Expected tensor with dimensions {expected_dims}, "
                            f"got {actual_dims}D tensor with shape {tuple(arg.shape)}. "
                            f"Argument position: {pos}"
                        )

                    # 验证最小尺寸（如果指定）
                    if min_dim_size is not None:
                        for i, size in enumerate(arg.shape):
                            if size < min_dim_size:
                                raise ValueError(
                                    f"[Validation] {func.__name__}(): "
                                    f"Dimension {i} of tensor has size {size}, "
                                    f"expected at least {min_dim_size}. "
                                    f"Shape: {tuple(arg.shape)}"
                                )

            return func(*args, **kwargs)

        return wrapper
    return decorator


def validate_tensor_range(min_value: Optional[float] = None, max_value: Optional[float] = None,
                         arg_positions: Optional[List[int]] = None):
    """
    装饰器：验证张量值范围

    Args:
        min_value: 允许的最小值（None表示不检查）
        max_value: 允许的最大值（None表示不检查）
        arg_positions: 要验证的参数位置列表
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not VALIDATION_ENABLED:
                return func(*args, **kwargs)

            positions = arg_positions if arg_positions is not None else [0]

            for pos in positions:
                if pos < len(args):
                    arg = args[pos]
                else:
                    continue

                if isinstance(arg, torch.Tensor):
                    # 检查最小值
                    if min_value is not None:
                        min_val = arg.min().item()
                        if min_val < min_value:
                            raise ValueError(
                                f"[Validation] {func.__name__}(): "
                                f"Tensor minimum value {min_val} < {min_value}. "
                                f"Argument position: {pos}"
                            )

                    # 检查最大值
                    if max_value is not None:
                        max_val = arg.max().item()
                        if max_val > max_value:
                            raise ValueError(
                                f"[Validation] {func.__name__}(): "
                                f"Tensor maximum value {max_val} > {max_value}. "
                                f"Argument position: {pos}"
                            )

            return func(*args, **kwargs)

        return wrapper
    return decorator


def validate_device_consistency(arg_positions: Optional[List[int]] = None):
    """
    装饰器：验证多个张量在同一设备上

    Args:
        arg_positions: 要验证的参数位置列表
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not VALIDATION_ENABLED:
                return func(*args, **kwargs)

            positions = arg_positions if arg_positions is not None else [0, 1]
            devices = []

            for pos in positions:
                if pos < len(args):
                    arg = args[pos]
                else:
                    continue

                if isinstance(arg, torch.Tensor):
                    devices.append(arg.device)

            # 检查所有设备是否一致
            if len(devices) > 1:
                first_device = devices[0]
                for i, device in enumerate(devices[1:], 1):
                    if device != first_device:
                        raise RuntimeError(
                            f"[Validation] {func.__name__}(): "
                            f"Device mismatch: arg[0] on {first_device}, "
                            f"arg[{i}] on {device}"
                        )

            return func(*args, **kwargs)

        return wrapper
    return decorator


def validate_batch_size(arg_positions: Optional[List[int]] = None):
    """
    装饰器：验证多个张量的batch size一致

    Args:
        arg_positions: 要验证的参数位置列表
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not VALIDATION_ENABLED:
                return func(*args, **kwargs)

            positions = arg_positions if arg_positions is not None else [0, 1]
            batch_sizes = []

            for pos in positions:
                if pos < len(args):
                    arg = args[pos]
                else:
                    continue

                if isinstance(arg, torch.Tensor) and arg.dim() >= 1:
                    batch_sizes.append(arg.size(0))

            # 检查所有batch size是否一致
            if len(batch_sizes) > 1:
                first_batch_size = batch_sizes[0]
                for i, batch_size in enumerate(batch_sizes[1:], 1):
                    if batch_size != first_batch_size:
                        raise ValueError(
                            f"[Validation] {func.__name__}(): "
                            f"Batch size mismatch: arg[0] has batch_size={first_batch_size}, "
                            f"arg[{i}] has batch_size={batch_size}"
                        )

            return func(*args, **kwargs)

        return wrapper
    return decorator


def validate_positive_integers(param_names: List[str]):
    """
    装饰器：验证指定参数为正整数

    Args:
        param_names: 要验证的参数名称列表
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not VALIDATION_ENABLED:
                return func(*args, **kwargs)

            # 检查kwargs中的参数
            for param_name in param_names:
                if param_name in kwargs:
                    value = kwargs[param_name]
                    if not isinstance(value, int) or value <= 0:
                        raise ValueError(
                            f"[Validation] {func.__name__}(): "
                            f"Parameter '{param_name}' must be a positive integer, "
                            f"got {value} (type: {type(value).__name__})"
                        )

            return func(*args, **kwargs)

        return wrapper
    return decorator


def validate_in_range(param_name: str, min_value: float, max_value: float):
    """
    装饰器：验证参数在指定范围内

    Args:
        param_name: 要验证的参数名称
        min_value: 最小值（包含）
        max_value: 最大值（包含）
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not VALIDATION_ENABLED:
                return func(*args, **kwargs)

            if param_name in kwargs:
                value = kwargs[param_name]
                if not (min_value <= value <= max_value):
                    raise ValueError(
                        f"[Validation] {func.__name__}(): "
                        f"Parameter '{param_name}' must be in [{min_value}, {max_value}], "
                        f"got {value}"
                    )

            return func(*args, **kwargs)

        return wrapper
    return decorator


def validate_video_shape(arg_position: int = 0, expected_channels: int = 3,
                        expected_min_size: int = 1):
    """
    装饰器：验证视频张量形状 (B, T, C, H, W)

    Args:
        arg_position: 视频张量的参数位置
        expected_channels: 期望的通道数（通常为3表示RGB）
        expected_min_size: 期望的最小空间尺寸
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not VALIDATION_ENABLED:
                return func(*args, **kwargs)

            if arg_position < len(args):
                video = args[arg_position]

                if isinstance(video, torch.Tensor):
                    # 验证维度
                    if video.dim() != 5:
                        raise ValueError(
                            f"[Validation] {func.__name__}(): "
                            f"Expected 5D video tensor (B, T, C, H, W), "
                            f"got {video.dim()}D tensor with shape {tuple(video.shape)}"
                        )

                    B, T, C, H, W = video.shape

                    # 验证通道数
                    if C != expected_channels:
                        raise ValueError(
                            f"[Validation] {func.__name__}(): "
                            f"Expected {expected_channels} channels, got {C}"
                        )

                    # 验证最小尺寸
                    if H < expected_min_size or W < expected_min_size:
                        raise ValueError(
                            f"[Validation] {func.__name__}(): "
                            f"Expected spatial dimensions >= {expected_min_size}, "
                            f"got H={H}, W={W}"
                        )

                    # 验证batch size和帧数为正数
                    if B <= 0 or T <= 0:
                        raise ValueError(
                            f"[Validation] {func.__name__}(): "
                            f"Batch size and frames must be positive, "
                            f"got B={B}, T={T}"
                        )

            return func(*args, **kwargs)

        return wrapper
    return decorator


def enable_validation():
    """启用输入验证"""
    global VALIDATION_ENABLED
    VALIDATION_ENABLED = True


def disable_validation():
    """禁用输入验证"""
    global VALIDATION_ENABLED
    VALIDATION_ENABLED = False


def is_validation_enabled() -> bool:
    """检查验证是否启用"""
    return VALIDATION_ENABLED
