from .video_safetybench import VideoSafetyBenchAdapter
from .videochatgpt import VideoChatGPTAdapter

REGISTRY: dict[str, type] = {
    "video_safetybench": VideoSafetyBenchAdapter,
    "videochatgpt":      VideoChatGPTAdapter,
}


def get_adapter(name: str):
    """Return an instantiated adapter for the given dataset name."""
    if name not in REGISTRY:
        raise ValueError(
            f"Unknown dataset: {name!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name]()
