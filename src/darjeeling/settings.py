from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Literal

from pydantic import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)
    _settings_yaml_path: ClassVar[Path | None] = None

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, validation_alias="OPENAI_BASE_URL")
    openai_model: str = Field(default="gpt-4.1-mini", validation_alias="OPENAI_MODEL")
    openai_max_retries: int = Field(default=3, validation_alias="OPENAI_MAX_RETRIES")
    openai_retry_base_delay_s: float = Field(
        default=1.0,
        validation_alias="OPENAI_RETRY_BASE_DELAY_S",
    )
    openai_retry_max_delay_s: float = Field(
        default=8.0,
        validation_alias="OPENAI_RETRY_MAX_DELAY_S",
    )
    openai_timeout_s: float = Field(default=60.0, validation_alias="OPENAI_TIMEOUT_S")
    teacher_prompt_version: str = Field(
        default="teacher-v1",
        validation_alias="TEACHER_PROMPT_VERSION",
    )
    teacher_max_tokens: int = Field(default=256, validation_alias="TEACHER_MAX_TOKENS")
    l4_proposal_mode: Literal["disabled", "live"] = Field(
        default="disabled",
        validation_alias="L4_PROPOSAL_MODE",
    )
    proposal_max_tokens: int = Field(default=1024, validation_alias="PROPOSAL_MAX_TOKENS")
    prompt_cache_retention: str = Field(
        default="24h",
        validation_alias="PROMPT_CACHE_RETENTION",
    )
    local_slm_model: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        validation_alias="LOCAL_SLM_MODEL",
    )
    local_slm_mode: Literal["disabled", "shadow", "guarded"] = Field(
        default="disabled",
        validation_alias="LOCAL_SLM_MODE",
    )
    local_slm_device_policy: Literal["auto", "cpu", "mps", "cuda"] = Field(
        default="auto",
        validation_alias="LOCAL_SLM_DEVICE_POLICY",
    )
    local_slm_max_new_tokens: int = Field(
        default=256,
        validation_alias="LOCAL_SLM_MAX_NEW_TOKENS",
    )
    local_slm_confidence_threshold: float = Field(
        default=0.70,
        validation_alias="LOCAL_SLM_CONFIDENCE_THRESHOLD",
    )
    local_slm_prompt_version: str = Field(
        default="l3-prompt-v1",
        validation_alias="LOCAL_SLM_PROMPT_VERSION",
    )
    l1_rust_crate_dir: Path = Field(
        default=Path("native/l1_programbank"),
        validation_alias="L1_RUST_CRATE_DIR",
    )
    l1_rust_binary: Path | None = Field(default=None, validation_alias="L1_RUST_BINARY")
    l1_worker_timeout_s: float = Field(default=5.0, validation_alias="L1_WORKER_TIMEOUT_S")
    l1_agent_mode: Literal["disabled", "dry-run", "codex-cli"] = Field(
        default="disabled",
        validation_alias="L1_AGENT_MODE",
    )
    l1_agent_codex_command: str = Field(default="codex", validation_alias="L1_AGENT_CODEX_COMMAND")
    l1_agent_model: str | None = Field(default=None, validation_alias="L1_AGENT_MODEL")
    l1_agent_timeout_s: float = Field(default=900.0, validation_alias="L1_AGENT_TIMEOUT_S")
    l1_agent_dry_run_patch: Path | None = Field(
        default=None,
        validation_alias="L1_AGENT_DRY_RUN_PATCH",
    )
    l1_agent_sandbox: Literal["workspace-write", "danger-full-access"] = Field(
        default="workspace-write",
        validation_alias="L1_AGENT_SANDBOX",
    )
    l1_agent_approval_policy: Literal["never", "on-request", "untrusted"] = Field(
        default="never",
        validation_alias="L1_AGENT_APPROVAL_POLICY",
    )

    l1_min_precision: float = 0.98
    l2_enabled: bool = Field(default=True, validation_alias="L2_ENABLED")
    l2_guard_mode: Literal["learned", "always_accept"] = Field(
        default="learned",
        validation_alias="L2_GUARD_MODE",
    )
    l2_intent_model_family: Literal["sgd_logreg", "mlp"] = Field(
        default="sgd_logreg",
        validation_alias="L2_INTENT_MODEL_FAMILY",
    )
    l2_slot_model_family: Literal["token_sgd", "none"] = Field(
        default="token_sgd",
        validation_alias="L2_SLOT_MODEL_FAMILY",
    )
    l2_training_scope: Literal["teacher_train", "lower_miss"] = Field(
        default="teacher_train",
        validation_alias="L2_TRAINING_SCOPE",
    )
    l2_agent_mode: Literal["disabled", "dry-run", "codex-cli"] = Field(
        default="disabled",
        validation_alias="L2_AGENT_MODE",
    )
    l2_agent_codex_command: str = Field(default="codex", validation_alias="L2_AGENT_CODEX_COMMAND")
    l2_agent_model: str | None = Field(default=None, validation_alias="L2_AGENT_MODEL")
    l2_agent_timeout_s: float = Field(default=900.0, validation_alias="L2_AGENT_TIMEOUT_S")
    l2_agent_dry_run_patch: Path | None = Field(
        default=None,
        validation_alias="L2_AGENT_DRY_RUN_PATCH",
    )
    l2_agent_sandbox: Literal["workspace-write", "danger-full-access"] = Field(
        default="workspace-write",
        validation_alias="L2_AGENT_SANDBOX",
    )
    l2_agent_approval_policy: Literal["never", "on-request", "untrusted"] = Field(
        default="never",
        validation_alias="L2_AGENT_APPROVAL_POLICY",
    )
    l2_agent_run_validation: bool = Field(
        default=True,
        validation_alias="L2_AGENT_RUN_VALIDATION",
    )
    l2_frame_source: Literal["student", "retrieval"] = Field(
        default="retrieval",
        validation_alias="L2_FRAME_SOURCE",
    )
    l2_max_features: int = Field(default=50_000, validation_alias="L2_MAX_FEATURES")
    l2_max_iter: int = Field(default=1000, validation_alias="L2_MAX_ITER")
    l2_mlp_hidden_layer_sizes: tuple[int, ...] = Field(
        default=(64,),
        validation_alias="L2_MLP_HIDDEN_LAYER_SIZES",
    )
    l2_mlp_alpha: float = Field(default=0.0001, validation_alias="L2_MLP_ALPHA")
    l2_mlp_early_stopping: bool = Field(
        default=False,
        validation_alias="L2_MLP_EARLY_STOPPING",
    )
    l2_tuning_mode: Literal["disabled", "optuna"] = Field(
        default="disabled",
        validation_alias="L2_TUNING_MODE",
    )
    l2_tuning_trials: int = Field(default=16, validation_alias="L2_TUNING_TRIALS")
    l2_tuning_min_examples: int = Field(
        default=200,
        validation_alias="L2_TUNING_MIN_EXAMPLES",
    )
    l2_tuning_timeout_s: float | None = Field(
        default=None,
        validation_alias="L2_TUNING_TIMEOUT_S",
    )
    l2_tuning_validation_fraction: float = Field(
        default=0.25,
        validation_alias="L2_TUNING_VALIDATION_FRACTION",
    )
    l2_tuning_split_policy: Literal["chronological", "stratified_random"] = Field(
        default="chronological",
        validation_alias="L2_TUNING_SPLIT_POLICY",
    )
    l2_tuning_search_space: Literal["compact", "wide"] = Field(
        default="compact",
        validation_alias="L2_TUNING_SEARCH_SPACE",
    )
    l2_tuning_latency_weight: float = Field(
        default=0.01,
        validation_alias="L2_TUNING_LATENCY_WEIGHT",
    )
    l2_min_guarded_accuracy: float = 0.93
    l2_max_wrong_accept_rate: float = 0.05
    l2_min_runtime_examples: int = Field(default=30, validation_alias="L2_MIN_RUNTIME_EXAMPLES")
    promotion_accuracy_epsilon: float = 0.02
    force_promote_artifacts: bool = Field(
        default=False,
        validation_alias="FORCE_PROMOTE_ARTIFACTS",
    )
    hard_buffer_max_cases: int = Field(default=100, validation_alias="HARD_BUFFER_MAX_CASES")
    settings_file: Path | None = None
    l0_cost_usd_per_request: float = Field(default=0.0, validation_alias="L0_COST_USD_PER_REQUEST")
    l1_cost_usd_per_request: float = Field(default=0.0, validation_alias="L1_COST_USD_PER_REQUEST")
    l2_cost_usd_per_request: float = Field(
        default=0.00005,
        validation_alias="L2_COST_USD_PER_REQUEST",
    )
    l3_cost_usd_per_request: float = Field(default=0.0, validation_alias="L3_COST_USD_PER_REQUEST")
    l4_default_cost_usd_per_request: float = Field(
        default=0.01,
        validation_alias="L4_DEFAULT_COST_USD_PER_REQUEST",
    )
    l4_input_usd_per_million: float = Field(
        default=0.40,
        validation_alias="L4_INPUT_USD_PER_MILLION",
    )
    l4_cached_input_usd_per_million: float = Field(
        default=0.10,
        validation_alias="L4_CACHED_INPUT_USD_PER_MILLION",
    )
    l4_output_usd_per_million: float = Field(
        default=1.60,
        validation_alias="L4_OUTPUT_USD_PER_MILLION",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        sources: list[PydanticBaseSettingsSource] = [
            init_settings,
            env_settings,
            dotenv_settings,
        ]
        if cls._settings_yaml_path is not None:
            sources.append(
                YamlConfigSettingsSource(
                    settings_cls,
                    yaml_file=cls._settings_yaml_path,
                )
            )
        sources.append(file_secret_settings)
        return tuple(sources)


def load_settings(settings_path: Path | None = None) -> Settings:
    explicit_settings_path = settings_path is not None
    effective_settings_path = settings_path
    if effective_settings_path is None:
        default_settings_path = Path("settings.yaml")
        if default_settings_path.exists():
            effective_settings_path = default_settings_path
    if effective_settings_path is not None:
        effective_settings_path = effective_settings_path.expanduser()
        if not effective_settings_path.exists():
            if explicit_settings_path:
                raise FileNotFoundError(f"settings file not found: {effective_settings_path}")
            effective_settings_path = None
        elif not effective_settings_path.is_file():
            raise FileNotFoundError(f"settings path is not a file: {effective_settings_path}")

    Settings._settings_yaml_path = effective_settings_path
    try:
        settings = Settings()
    finally:
        Settings._settings_yaml_path = None
    return settings.model_copy(update={"settings_file": effective_settings_path})
