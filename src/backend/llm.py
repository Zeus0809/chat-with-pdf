from huggingface_hub import hf_hub_download
from src.backend.hardware import get_total_ram

class ModelManager():

    _EMBED_MODEL_CONFIG = {
        'repo' : 'wp-airik/nomic-embed-text-v2-moe-Q8_0-GGUF',
        'filename' : 'nomic-embed-text-v2-moe-q8_0.gguf'
        # 'filesize' : 512 MB
    }
    _CHAT_MODEL_CONFIGS = { 8 : { 
                            'repo': 'lmstudio-community/Qwen3-4B-Instruct-2507-GGUF',
                            'filename': 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf',
                            # 'filesize': 2.5 GB
                        },
                   16 : {
                            'repo': 'MaziyarPanahi/gemma-3-12b-it-GGUF',
                            'filename': 'gemma-3-12b-it.Q4_K_M.gguf',
                            # 'filesize': 7.3 GB
                        },
                   32 : {
                            'repo': 'EasierAI/Mistral-Small-3-24B',
                            'filename': 'Mistral-Small-3-24B-Instruct-Q4_K_M.gguf',
                            # 'filesize': 14.3 GB
                        },
                   64 : {
                            'repo': 'bartowski/c4ai-command-r-v01-GGUF',
                            'filename': 'c4ai-command-r-v01-Q6_K.gguf',
                            # 'filesize': 28.7 GB
                        } }
    
    def __init__(self):
       # get memory tier first to know what model to download
       self._memory_tier = self._determine_memory_tier()
       # download models
       self.embed_model_path = self._download_embed_model()
       self.chat_model_path = self._download_chat_model()
       self.embed_model_name = self.embed_model_path.replace('.gguf', '')
       self.chat_model_name = self.chat_model_path.replace('.gguf', '')

    def _download_chat_model(self) -> None:
        """
        Download a model from HuggingFace Hub based on user's machine RAM capabilities.
        `hf_hub_download` already checks for existence, so we don't need to.
        """
        model_filename = self._CHAT_MODEL_CONFIGS[self._memory_tier]['filename']
        path = hf_hub_download(
            repo_id=self._CHAT_MODEL_CONFIGS[self._memory_tier]['repo'],
            filename=model_filename
        )
        return path

    def _download_embed_model(self) -> None:
        """
        Download the `Nomic Embed Text V2 MoE` embedding model.
        `hf_hub_download` already checks for existence, so we don't need to.
        """
        model_filename = self._EMBED_MODEL_CONFIG['filename']
        path = hf_hub_download(
            repo_id=self._EMBED_MODEL_CONFIG['repo'],
            filename=model_filename
        )
        return path

    def _determine_memory_tier(self) -> int:
        """
        Cover all possible RAM cases on user's machine.
        Returns:
            One of: 8, 16, 32, 64 (GB of RAM).
        These correspond to four differend chat model configs based on user's RAM capacity. See `ModelManager.CHAT_MODEL_CONFIGS`
        """
        user_ram = get_total_ram()
        assert isinstance(user_ram, int), f"user_ram should be an integer, instead got {type(user_ram)}" 
        if user_ram < 16:
            return 8
        elif user_ram >= 16 and user_ram < 32:
            return 16
        elif user_ram >= 32 and user_ram < 64:
            return 32
        else:
            return 64


