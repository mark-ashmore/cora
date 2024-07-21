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
ENTITIES_PATH = Path(__file__).parent.parent / "custom_entities"

# User configs
CONFIGS_PATH = Path(__file__).parent.parent / "configs"
CHAT_HISTORY = CONFIGS_PATH / "chat_history.pkl"
PROMPTS_PATH = CONFIGS_PATH / "prompts"
SETTINGS_PATH = CONFIGS_PATH / "settings.json"
CUSTOM_SETTINGS_PATH = CONFIGS_PATH / "custom_settings.json"

# Logs
LOGS_PATH = Path(__file__).parent.parent / "logs"
ASSISTANT_MAIN_LOG = LOGS_PATH / "assistant_main.log"
MAIN_PIPELINE_LOG = LOGS_PATH / "main_pipeline.log"
MODEL_REPORT_PATH = LOGS_PATH / "model_report.csv"

# Actions
ACTIONS_PATH = Path(__file__).parent.parent / "agent_actions"
LIGHT_APP_NAMES = ACTIONS_PATH / "light_app_names.json"
