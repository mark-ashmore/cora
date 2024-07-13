"""Common agent paths."""

from pathlib import Path

# Audio
AUDIO_PATH = Path(__file__).parent.parent / "audio"
AGENT_RESPONSE_PATH = AUDIO_PATH / "agent_response.mp3"
GREETING_SOUND = AUDIO_PATH / "greeting.mp3"
WAKE_SOUND = AUDIO_PATH / "wake_up.mp3"

# NLU
NLU_PATH = Path(__file__).parent.parent / "main_pipeline"
BERT_PATH = NLU_PATH / "bert_model"
ENTITY_MODEL_PATH = NLU_PATH / "entity_model"
LABELS_PATH = NLU_PATH / "labels.csv"
TOKENIZER_PATH = NLU_PATH / "bert_tokenizer"

# Model Training
MODEL_TRAINING_PATH = Path(__file__).parent.parent / "model_training"

# User configs
CONFIGS_PATH = Path(__file__).parent.parent / "configs"
CHAT_HISTORY = CONFIGS_PATH / "chat_history.pkl"
PROMPTS_PATH = CONFIGS_PATH / "prompts"
SETTINGS_PATH = CONFIGS_PATH / "settings.json"
CUSTOM_SETTINGS_PATH = CONFIGS_PATH / "custom_settings.json"
