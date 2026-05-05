"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: Apache License 2.0
 For full license text, see the LICENSE file in the repo root or https://www.apache.org/licenses/LICENSE-2.0
"""


import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "empty_key")
BING_SUBSCRIPTION_KEY = os.environ.get("BING_SUBSCRIPTION_KEY", "empty_key")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "empty_key")
