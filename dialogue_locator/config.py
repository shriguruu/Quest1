from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOCATOR_", env_file=".env")

    # --- ASR ---
    # base.en is the default because it is fast enough to be usable on CPU
    # (~46x realtime) and accurate enough for this task: the matcher's fuzzy
    # and phonetic tiers exist precisely to absorb small-model transcription
    # error. Raise to small.en / medium.en / large-v3 for difficult audio --
    # heavy accents, noise, overlapping speakers -- or use the multilingual
    # names (base, small, large-v3) for non-English media.
    whisper_model: str = "base.en"
    whisper_compute_type: str = "int8_float16"
    device: str = "cuda"
    # Used when CUDA initialisation fails and we fall back to CPU.
    cpu_compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 5
    min_silence_duration_ms: int = 500

    # Coarse-to-fine: a small model scans the whole file to find the region,
    # then a larger model re-transcribes a narrow window around it for
    # accurate word timestamps.  Both are English-only by default because
    # the ".en" variants are meaningfully more accurate at these sizes; set
    # them to the multilingual names ("base", "small", "medium") for
    # non-English media.
    coarse_model: str = "base.en"
    fine_model: str = "small.en"

    # --- Paths ---
    cache_dir: str = ".cache"
    output_dir: str = "output"

    # --- Matching / confidence ---
    confident_threshold: float = 85.0
    uncertain_threshold: float = 65.0
    max_alternates: int = 5

    # --- Alignment ---
    # Half-width of the window re-transcribed by the fine model, in seconds.
    fine_window_s: float = 12.0
    align_window_s: float = 3.0

    # --- Visual / OCR ---
    ocr_sample_fps: float = 2.0
    ocr_band_top: float = 0.55  # OCR only the bottom 45% of the frame
    ocr_refine_step_s: float = 0.25


settings = Settings()
