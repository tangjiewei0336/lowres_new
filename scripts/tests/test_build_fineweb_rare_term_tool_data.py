from scripts.dictionary.build_fineweb_rare_term_tool_data import (
    build_prompt,
    default_max_token_len,
    default_min_token_len,
    is_noise_token,
    normalize_senses,
    tokenize,
)


def test_prompt_requests_sense_level_examples() -> None:
    messages = build_prompt(
        term="claim",
        lang="eng_Latn",
        frequency=1,
        contexts=["The claim was challenged."],
    )
    text = "\n".join(m["content"] for m in messages)
    assert "`senses`" in text
    assert "Each distinct meaning" in text
    assert "example_sentence" in text


def test_normalize_senses_from_new_schema() -> None:
    senses = normalize_senses(
        {
            "senses": [
                {
                    "definition": "A statement presented as true.",
                    "usage_note": "Often needs evidence.",
                    "example_sentence": "The claim requires proof.",
                }
            ]
        }
    )
    assert senses == [
        {
            "definition": "A statement presented as true.",
            "usage_note": "Often needs evidence.",
            "example_sentence": "The claim requires proof.",
        }
    ]


def test_normalize_senses_from_legacy_schema() -> None:
    senses = normalize_senses(
        {
            "definition": "A statement presented as true.",
            "usage_note": "Often needs evidence.",
            "example_sentences": ["The claim requires proof."],
        }
    )
    assert senses[0]["example_sentence"] == "The claim requires proof."


def test_latin_tokenizer_filters_stopwords_and_noise() -> None:
    toks = tokenize(
        "The quokkaesque mural appeared at https://example.com in 2026.",
        lang="eng_Latn",
        min_token_len=default_min_token_len("eng_Latn"),
        max_token_len=default_max_token_len("eng_Latn"),
    )
    assert "quokkaesque" in toks
    assert "the" not in toks
    assert "https" not in toks


def test_chinese_tokenizer_uses_cjk_ngrams() -> None:
    toks = tokenize(
        "斯坦福大学医学院宣布新型诊断工具。",
        lang="zho_Hans",
        min_token_len=default_min_token_len("zho_Hans"),
        max_token_len=default_max_token_len("zho_Hans"),
    )
    assert "斯坦福" in toks
    assert "诊断工具" in toks


def test_thai_tokenizer_returns_terms() -> None:
    toks = tokenize(
        "ภาษาไทยมีการตัดคำที่น่าสนใจ",
        lang="tha_Thai",
        min_token_len=default_min_token_len("tha_Thai"),
        max_token_len=default_max_token_len("tha_Thai"),
    )
    assert toks


def test_noise_filter_rejects_digits_and_repeats() -> None:
    assert is_noise_token("abc123", lang="eng_Latn", min_token_len=4, max_token_len=40)
    assert is_noise_token("aaaa", lang="eng_Latn", min_token_len=4, max_token_len=40)
