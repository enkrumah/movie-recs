# src/client.py
"""Shared OpenAI client singleton."""
from __future__ import annotations
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

_client: OpenAI | None = None


def get_openai_client() -> OpenAI:
    """
    Return a singleton OpenAI client.
    
    Loads API key from:
    1. .env file (local development)
    2. Environment variable
    3. Streamlit secrets (cloud deployment)
    """
    global _client
    if _client is None:
        load_dotenv(Path(".env"))
        api_key = os.getenv("OPENAI_API_KEY")
        
        # Fallback to Streamlit secrets if available
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("OPENAI_API_KEY")
            except Exception:
                pass
        
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY not set. Add it to .env or Streamlit secrets."
            )
        _client = OpenAI(api_key=api_key)
    return _client

