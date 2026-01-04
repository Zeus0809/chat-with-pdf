import platform, os, psutil

def _is_apple_silicon() -> bool:
    """Detect if user's machine is an Apple Silicon Mac to use Metal acceleration"""
    return platform.system() == "Darwin" and platform.machine() == "arm64"

def get_free_ram() -> int:
    """
    Returns:
        The amount of unused, free RAM available on user's machine in GB, rounded to 1 decimal space.
    """
    ram = psutil.virtual_memory().available / (1024 ** 3)
    return round(ram, 1)

def get_total_ram() -> int:
    """
    Returns:
        Total RAM capacity of user's machine.
    """
    return psutil.virtual_memory().total // (1024 ** 3)

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

    

    

