"""
Base Engine Module
所有 AI 引擎的抽象基类，定义统一接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, AsyncIterator, Union
import base64
from pathlib import Path


class MessageRole(str, Enum):
    """消息角色"""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class ImageContent:
    """图像内容"""
    # 可以是 base64 编码、URL 或本地文件路径
    data: str
    mime_type: str = "image/png"
    is_url: bool = False
    is_base64: bool = False

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "ImageContent":
        """从文件加载图像"""
        path = Path(path)
        suffix = path.suffix.lower()
        mime_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        mime_type = mime_map.get(suffix, "image/png")

        with open(path, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")

        return cls(data=data, mime_type=mime_type, is_base64=True)

    @classmethod
    def from_url(cls, url: str, mime_type: str = "image/png") -> "ImageContent":
        """从 URL 创建图像引用"""
        return cls(data=url, mime_type=mime_type, is_url=True)

    @classmethod
    def from_base64(cls, data: str, mime_type: str = "image/png") -> "ImageContent":
        """从 base64 数据创建"""
        return cls(data=data, mime_type=mime_type, is_base64=True)


@dataclass
class Message:
    """对话消息"""
    role: MessageRole
    content: str
    images: list[ImageContent] = field(default_factory=list)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str, images: list[ImageContent] = None) -> "Message":
        return cls(role=MessageRole.USER, content=content, images=images or [])

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(role=MessageRole.ASSISTANT, content=content)


@dataclass
class GenerationConfig:
    """生成配置"""
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.95
    top_k: int = 40
    stop_sequences: list[str] = field(default_factory=list)
    # Seed / OpenAI-compatible extras (ignored by engines that do not support them)
    thinking: Optional[bool] = None            # None = engine default
    response_format: Optional[dict] = None     # e.g. {"type": "json_schema", "json_schema": {...}}
    model: Optional[str] = None                # per-call model override
    reasoning_effort: Optional[str] = None     # with thinking: low | medium | high (None = engine default)
    prompt_cache_key: Optional[str] = None     # routes the calls of one job to the provider's cache of their shared prefix
    cache_prefix: bool = False                 # the first message is a long prefix many calls share (the paper): cache it explicitly


@dataclass
class ImageGenerationConfig:
    """图像生成配置"""
    width: int = 1024
    height: int = 1536
    num_images: int = 1
    style: str = "full_color_manga"
    negative_prompt: str = ""                  # not supported by Seedream (kept for Gemini engines)
    temperature: float = 0.3                   # not supported by Seedream (kept for Gemini engines)
    max_tokens: int = 4096
    # provider-specific knobs, e.g. Seedream: {"fast": bool, "output_format": "png"|"jpeg", "watermark": bool, "size": "2K"}
    extra: dict = field(default_factory=dict)


@dataclass
class TextResponse:
    """文本响应"""
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    finish_reason: str = ""


@dataclass
class ImageResponse:
    """图像响应"""
    images: list[ImageContent]
    model: str
    prompt: str
    revised_prompt: Optional[str] = None


class BaseEngine(ABC):
    """AI 引擎基类"""

    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model

    @property
    @abstractmethod
    def name(self) -> str:
        """引擎名称"""
        pass

    @property
    @abstractmethod
    def supports_image_generation(self) -> bool:
        """是否支持图像生成"""
        pass

    @property
    @abstractmethod
    def supports_vision(self) -> bool:
        """是否支持视觉理解（图像输入）"""
        pass

    @abstractmethod
    async def generate_text(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None
    ) -> TextResponse:
        """生成文本"""
        pass

    @abstractmethod
    async def generate_text_stream(
        self,
        messages: list[Message],
        config: Optional[GenerationConfig] = None
    ) -> AsyncIterator[str]:
        """流式生成文本"""
        pass

    async def generate_image(
        self,
        prompt: str,
        config: Optional[ImageGenerationConfig] = None,
        reference_images: list[ImageContent] = None
    ) -> ImageResponse:
        """生成图像（子类可选实现）"""
        raise NotImplementedError(f"{self.name} does not support image generation")

    async def close(self) -> None:
        """关闭连接/清理资源"""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
