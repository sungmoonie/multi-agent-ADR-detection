"""
Utility functions for the ADR detection pipeline.

Provides binary classification of LLM JSON responses (ADR present / absent).
"""

import pandas as pd
import ast
import json


def classify_certainty(x):
    """Convert a side_effect column value to binary: 0 (no ADR) or 1 (ADR present)."""

    # 1. List or dict input (handle first)
    if isinstance(x, (list, dict)):
        if len(x) == 0:
            return 0
        s_obj = str(x)
        if "No Side Effect" in s_obj or "Parsing Error" in s_obj:
            return 0
        return 1

    # 2. Other types (string, NaN, numeric, etc.)
    try:
        if pd.isna(x):
            return 0
    except Exception:
        pass

    s = str(x).strip()

    if s == "" or s.lower() == "nan":
        return 0

    if "No Side Effect" in s or "Parsing Error" in s:
        return 0

    # 3. Attempt to parse string-form list/JSON (e.g., "['symptom']")
    try:
        parsed = None
        try:
            parsed = json.loads(s)
        except Exception:
            try:
                parsed = ast.literal_eval(s)
            except Exception:
                pass

        if isinstance(parsed, list):
            return 1 if len(parsed) > 0 else 0
        elif isinstance(parsed, dict):
            return 1 if len(parsed) > 0 else 0
    except Exception:
        pass

    # 4. Non-empty text without "No Side Effect" → treat as ADR present
    return 1
