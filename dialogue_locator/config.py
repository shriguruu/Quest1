from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOCATOR_", env_file=".env")

    whisper_model: str = "large-v3"
    whisper_compute_type: str = "int8_float16"
    device: str = "cuda"
    cache_dir: str = ".cache"
    output_dir: str = "output"
    confident_threshold: float = 85.0
    uncertain_threshold: float = 65.0
    align_window_s: float = 3.0
    ocr_sample_fps: float = 2.0

settings = Settings()
