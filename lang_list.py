# Languages dict
LANGUAGE_NAME_TO_CODE = {
    "العربية": "ar_AR",
    "Čeština": "cs_CZ",
    "Deutsch": "de_DE",
    "English": "en_XX",
    "Español": "es_XX",
    "Eesti": "et_EE",
    "Suomi": "fi_FI",
    "Français": "fr_XX",
    "ગુજરાતી": "gu_IN",
    "हिन्दी": "hi_IN",
    "Italiano": "it_IT",
    "日本語": "ja_XX",
    "Қазақ": "kk_KZ",
    "한국어": "ko_KR",
    "Lietuvių": "lt_LT",
    "Latviešu": "lv_LV",
    "ဗမာ": "my_MM",
    "नेपाली": "ne_NP",
    "Nederlands": "nl_XX",
    "Română": "ro_RO",
    "Русский": "ru_RU",
    "සිංහල": "si_LK",
    "Türkçe": "tr_TR",
    "Tiếng Việt": "vi_VN",
    "中文": "zh_CN",
    "Afrikaans": "af_ZA",
    "Azərbaycan": "az_AZ",
    "বাংলা": "bn_IN",
    "فارسی": "fa_IR",
    "עברית": "he_IL",
    "Hrvatski": "hr_HR",
    "Indonesia": "id_ID",
    "ქართული": "ka_GE",
    "ខ្មែរ": "km_KH",
    "Македонски": "mk_MK",
    "മലയാളം": "ml_IN",
    "Монгол": "mn_MN",
    "मराठी": "mr_IN",
    "Polski": "pl_PL",
    "پښتو": "ps_AF",
    "Português": "pt_XX",
    "Svenska": "sv_SE",
    "Kiswahili": "sw_KE",
    "தமிழ்": "ta_IN",
    "తెలుగు": "te_IN",
    "ไทย": "th_TH",
    "Tagalog": "tl_XX",
    "Українська": "uk_UA",
    "اردو": "ur_PK",
    "isiXhosa": "xh_ZA",
    "Galego": "gl_ES",
    "Slovenščina": "sl_SI"
}

# Whisper languages dict
WHISPER_LANGUAGES = {
    "en": "english",
    "zh": "chinese",
    "de": "german",
    "es": "spanish",
    "ru": "russian",
    "ko": "korean",
    "fr": "french",
    "ja": "japanese",
    "pt": "portuguese",
    "tr": "turkish",
    "pl": "polish",
    "ca": "catalan",
    "nl": "dutch",
    "ar": "arabic",
    "sv": "swedish",
    "it": "italian",
    "id": "indonesian",
    "hi": "hindi",
    "fi": "finnish",
    "vi": "vietnamese",
    "he": "hebrew",
    "uk": "ukrainian",
    "el": "greek",
    "ms": "malay",
    "cs": "czech",
    "ro": "romanian",
    "da": "danish",
    "hu": "hungarian",
    "ta": "tamil",
    "no": "norwegian",
    "th": "thai",
    "ur": "urdu",
    "hr": "croatian",
    "bg": "bulgarian",
    "lt": "lithuanian",
    "la": "latin",
    "mi": "maori",
    "ml": "malayalam",
    "cy": "welsh",
    "sk": "slovak",
    "te": "telugu",
    "fa": "persian",
    "lv": "latvian",
    "bn": "bengali",
    "sr": "serbian",
    "az": "azerbaijani",
    "sl": "slovenian",
    "kn": "kannada",
    "et": "estonian",
    "mk": "macedonian",
    "br": "breton",
    "eu": "basque",
    "is": "icelandic",
    "hy": "armenian",
    "ne": "nepali",
    "mn": "mongolian",
    "bs": "bosnian",
    "kk": "kazakh",
    "sq": "albanian",
    "sw": "swahili",
    "gl": "galician",
    "mr": "marathi",
    "pa": "punjabi",
    "si": "sinhala",
    "km": "khmer",
    "sn": "shona",
    "yo": "yoruba",
    "so": "somali",
    "af": "afrikaans",
    "oc": "occitan",
    "ka": "georgian",
    "be": "belarusian",
    "tg": "tajik",
    "sd": "sindhi",
    "gu": "gujarati",
    "am": "amharic",
    "yi": "yiddish",
    "lo": "lao",
    "uz": "uzbek",
    "fo": "faroese",
    "ht": "haitian creole",
    "ps": "pashto",
    "tk": "turkmen",
    "nn": "nynorsk",
    "mt": "maltese",
    "sa": "sanskrit",
    "lb": "luxembourgish",
    "my": "myanmar",
    "bo": "tibetan",
    "tl": "tagalog",
    "mg": "malagasy",
    "as": "assamese",
    "tt": "tatar",
    "haw": "hawaiian",
    "ln": "lingala",
    "ha": "hausa",
    "ba": "bashkir",
    "jw": "javanese",
    "su": "sundanese",
}

def union_language_dict():
    # Create a dictionary to store the language codes
    language_dict = {}
    # Iterate over the LANGUAGE_NAME_TO_CODE dictionary
    for language_name, language_code in LANGUAGE_NAME_TO_CODE.items():
        # Extract the language code (the first two characters before the underscore)
        lang_code = language_code.split('_')[0].lower()
        
        # Check if the language code is present in WHISPER_LANGUAGES
        if lang_code in WHISPER_LANGUAGES:
            # Construct the entry for the resulting dictionary
            language_dict[language_name] = {
                "transcriber": lang_code,
                "translator": language_code
            }
    return language_dict