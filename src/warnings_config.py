# src/warnings_config.py

import warnings

from langsmith.utils import LangSmithMissingAPIKeyWarning

warnings.filterwarnings("ignore", category=LangSmithMissingAPIKeyWarning)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message=".*agent.agent_builder.*"
)
