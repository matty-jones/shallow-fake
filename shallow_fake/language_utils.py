"""Language and region utilities for Piper model naming and metadata."""

from typing import Dict, Optional, Tuple


# Language metadata mapping
# Format: (language_code, region) -> (name_native, name_english, country_english)
LANGUAGE_METADATA: Dict[Tuple[str, str], Dict[str, str]] = {
    ("en", "GB"): {
        "name_native": "English",
        "name_english": "English",
        "country_english": "United Kingdom",
    },
    ("en", "US"): {
        "name_native": "English",
        "name_english": "English",
        "country_english": "United States",
    },
    ("en", "AU"): {
        "name_native": "English",
        "name_english": "English",
        "country_english": "Australia",
    },
    ("en", "CA"): {
        "name_native": "English",
        "name_english": "English",
        "country_english": "Canada",
    },
    ("en", "IE"): {
        "name_native": "English",
        "name_english": "English",
        "country_english": "Ireland",
    },
    ("en", "NZ"): {
        "name_native": "English",
        "name_english": "English",
        "country_english": "New Zealand",
    },
    ("es", "ES"): {
        "name_native": "Español",
        "name_english": "Spanish",
        "country_english": "Spain",
    },
    ("es", "MX"): {
        "name_native": "Español",
        "name_english": "Spanish",
        "country_english": "Mexico",
    },
    ("fr", "FR"): {
        "name_native": "Français",
        "name_english": "French",
        "country_english": "France",
    },
    ("de", "DE"): {
        "name_native": "Deutsch",
        "name_english": "German",
        "country_english": "Germany",
    },
    ("it", "IT"): {
        "name_native": "Italiano",
        "name_english": "Italian",
        "country_english": "Italy",
    },
    ("pt", "BR"): {
        "name_native": "Português",
        "name_english": "Portuguese",
        "country_english": "Brazil",
    },
    ("pt", "PT"): {
        "name_native": "Português",
        "name_english": "Portuguese",
        "country_english": "Portugal",
    },
    ("nl", "NL"): {
        "name_native": "Nederlands",
        "name_english": "Dutch",
        "country_english": "Netherlands",
    },
    ("pl", "PL"): {
        "name_native": "Polski",
        "name_english": "Polish",
        "country_english": "Poland",
    },
    ("ru", "RU"): {
        "name_native": "Русский",
        "name_english": "Russian",
        "country_english": "Russia",
    },
    ("ja", "JP"): {
        "name_native": "日本語",
        "name_english": "Japanese",
        "country_english": "Japan",
    },
    ("zh", "CN"): {
        "name_native": "中文",
        "name_english": "Chinese",
        "country_english": "China",
    },
    ("ko", "KR"): {
        "name_native": "한국어",
        "name_english": "Korean",
        "country_english": "South Korea",
    },
}


def parse_language_code(language: str) -> Tuple[str, str]:
    """
    Parse language code in format 'en_GB' or 'en-US' into (language, region).
    
    Args:
        language: Language code in format 'en_GB', 'en-US', 'en', etc.
        
    Returns:
        Tuple of (language_code, region). If region is not specified, defaults to 'US' for 'en', otherwise uses language code as region.
    """
    # Normalize separators
    normalized = language.replace("-", "_").upper()
    
    if "_" in normalized:
        parts = normalized.split("_", 1)
        lang_code = parts[0].lower()
        region = parts[1]
        return (lang_code, region)
    else:
        # No region specified, use defaults
        lang_code = normalized.lower()
        # Default region for common languages
        if lang_code == "en":
            region = "US"
        else:
            # Use language code as region if no region specified
            region = normalized
        return (lang_code, region)


def get_language_metadata(language_code: str, region: str) -> Dict[str, str]:
    """
    Get language metadata for a given language code and region.
    
    Args:
        language_code: Language code (e.g., 'en')
        region: Region code (e.g., 'GB')
        
    Returns:
        Dictionary with name_native, name_english, and country_english.
        Returns defaults if not found in mapping.
    """
    key = (language_code.lower(), region.upper())
    
    if key in LANGUAGE_METADATA:
        return LANGUAGE_METADATA[key].copy()
    
    # Default fallback
    return {
        "name_native": language_code.capitalize(),
        "name_english": language_code.capitalize(),
        "country_english": region,
    }


def format_model_name(language_code: str, region: str, name: str, quality: str) -> str:
    """
    Format model name according to Piper convention: <language>_<REGION>-<name>-<quality>
    
    Args:
        language_code: Language code (e.g., 'en')
        region: Region code (e.g., 'GB')
        name: Model name (e.g., 'matty')
        quality: Quality level (e.g., 'high')
        
    Returns:
        Formatted model name (e.g., 'en_GB-matty-high')
    """
    return f"{language_code.lower()}_{region.upper()}-{name}-{quality}"


def convert_language_for_phoneme(language: str) -> str:
    """
    Convert language code from model format to phoneme checking format.
    
    Converts formats like:
    - 'en_GB' -> 'en-gb'
    - 'en-US' -> 'en-us'
    - 'en' -> 'en'
    
    Args:
        language: Language code in format 'en_GB', 'en-US', etc.
        
    Returns:
        Language code in hyphenated lowercase format for phoneme checking (e.g., 'en-gb')
    """
    # Normalize separators to underscore first
    normalized = language.replace("-", "_")
    
    if "_" in normalized:
        parts = normalized.split("_", 1)
        lang_code = parts[0].lower()
        region = parts[1].lower()
        return f"{lang_code}-{region}"
    else:
        # No region specified, return as-is (lowercase)
        return normalized.lower()


# Espeak voice mapping
# Maps (language_code, region) to espeak voice names
# Based on valid espeak voices from espeak-ng --voices
ESPEAK_VOICE_MAPPING: Dict[Tuple[str, str], str] = {
    # English variants
    ("en", "GB"): "en-gb-x-rp",  # Received Pronunciation (preferred for GB)
    ("en", "US"): "en-us",
    ("en", "AU"): "en",  # Fallback to generic English
    ("en", "CA"): "en",  # Fallback to generic English
    ("en", "IE"): "en",  # Fallback to generic English
    ("en", "NZ"): "en",  # Fallback to generic English
    # Spanish variants
    ("es", "ES"): "es",  # Spanish (Spain)
    ("es", "MX"): "es-419",  # Spanish (Latin America)
    # French variants
    ("fr", "FR"): "fr-fr",
    ("fr", "BE"): "fr-be",
    ("fr", "CH"): "fr-ch",
    # Portuguese variants
    ("pt", "PT"): "pt",  # Portuguese (Portugal)
    ("pt", "BR"): "pt-br",  # Portuguese (Brazil)
    # Other languages (use language code directly)
    ("de", "DE"): "de",
    ("it", "IT"): "it",
    ("nl", "NL"): "nl",
    ("pl", "PL"): "pl",
    ("ru", "RU"): "ru",
    ("ru", "LV"): "ru-lv",
    ("ja", "JP"): "ja",
    ("zh", "CN"): "cmn",  # Chinese (Mandarin)
    ("ko", "KR"): "ko",
    # Add more mappings as needed
}


def get_espeak_voice(language_code: str, region: str) -> str:
    """
    Get espeak voice name for a given language code and region.
    
    Maps language codes to valid espeak voices based on the espeak-ng voice list.
    For en_GB, returns 'en-gb-x-rp' (Received Pronunciation) as preferred.
    
    Args:
        language_code: Language code (e.g., 'en')
        region: Region code (e.g., 'GB')
        
    Returns:
        Espeak voice name (e.g., 'en-gb-x-rp', 'en-us', 'de', etc.)
    """
    key = (language_code.lower(), region.upper())
    
    if key in ESPEAK_VOICE_MAPPING:
        return ESPEAK_VOICE_MAPPING[key]
    
    # Fallback: use language code as espeak voice (works for many languages)
    # Most single-language codes like 'de', 'fr', 'it' work directly
    # This handles cases where the language code matches an espeak voice directly
    return language_code.lower()

