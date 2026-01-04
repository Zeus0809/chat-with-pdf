import platform, os

def _is_apple_silicon() -> bool:
    """Detect if user's machine is an Apple Silicon Mac to use Metal acceleration"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def get_optimal_config() -> dict:
    """
    Determine optimal LlamaCPP model_kwargs based on user's hardware capabilities.
    Returns:
        A dict with `n_gpu_layers` and `n_threads` values.
    """
    if _is_apple_silicon():
        return { 'n_gpu_layers': -1, 'n_threads': 1 } # all layers on Metal, 1 CPU thread for other LLM ops
    else:
        return { 'n_gpu_layers': 0, 'n_threads': os.cpu_count() } # CPU inference, use all threads

    

    

