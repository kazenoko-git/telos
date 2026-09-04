from .transformer import TelosConfig, TelosTransformer

def __getattr__(name: str):
    if name == "MLXTelosTransformer":
        try:
            from .mlx_transformer import MLXTelosTransformer
            return MLXTelosTransformer
        except ImportError as err:
            raise ImportError(
                "MLXTelosTransformer requires 'mlx', which is not available in this environment. "
                "Install it on Apple Silicon via `pip install 'telos[mlx]'`."
            ) from err
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["TelosConfig", "TelosTransformer", "MLXTelosTransformer"]
