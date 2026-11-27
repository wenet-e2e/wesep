# Copyright (c) 2026 Ke Zhang (kylezhang1118@gmail.com)
# SPDX-License-Identifier: Apache-2.0
"""CPU text tokenization shared by cue workers, preprocessing and the CLI."""

import os
import string
import unicodedata


class DAEPhonemeTokenizer:
    """Convert English text to the phoneme IDs used by the DAE KCE."""

    def __init__(self, phoneme_map, lexicon=None):
        self.g2p = None
        self.phoneme_to_id = {}
        with open(phoneme_map, encoding="utf-8") as stream:
            for line in stream:
                fields = line.strip().split()
                if len(fields) == 2:
                    self.phoneme_to_id[fields[0]] = int(fields[1])

        self.lexicon = {}
        if lexicon and os.path.isfile(lexicon):
            with open(lexicon, encoding="utf-8") as stream:
                for line in stream:
                    fields = line.strip().split(maxsplit=1)
                    if len(fields) == 2:
                        self.lexicon[fields[0]] = [
                            int(value) for value in fields[1].split()
                        ]

    @staticmethod
    def normalize(text):
        """Normalize English text while preserving word apostrophes."""
        text = (text or "").strip().lower()
        for apostrophe in ("\u2019", "\u2018", "\u0060", "\u00b4"):
            text = text.replace(apostrophe, "'")
        cleaned = []
        for char in text:
            if char == "'":
                cleaned.append(char)
            elif char in string.punctuation or unicodedata.category(
                    char).startswith("P"):
                cleaned.append(" ")
            else:
                cleaned.append(char)
        return " ".join("".join(cleaned).split())

    def encode_words(self, text):
        """Return normalized words and one phoneme-ID sequence per word."""
        normalized = self.normalize(text)
        words = normalized.split()
        if not words:
            raise ValueError("Text is empty after DAE normalization")
        labels = []
        for word in words:
            ids = self.lexicon.get(word)
            if ids is None:
                if self.g2p is None:
                    from g2p_en import G2p
                    self.g2p = G2p()
                phones = self.g2p(word)
                ids = [
                    self.phoneme_to_id[phone] for phone in phones
                    if phone in self.phoneme_to_id
                ]
                self.lexicon[word] = ids
            if not ids:
                raise ValueError(f"No DAE phoneme IDs generated for: {word}")
            labels.append(ids)
        return normalized, words, labels

    def encode(self, text):
        """Return one flattened phoneme-ID sequence for direct inference."""
        _, _, labels = self.encode_words(text)
        return [value for word in labels for value in word]

    def encode_word_groups(self, words):
        """Keep one phoneme group per original word, including compound words."""
        return [self.encode(word) for word in words]


def build_text_tokenizer(config):
    """Construct the configured CPU tokenizer without loading a cue encoder."""
    if config.get("type") != "dae_phoneme":
        raise ValueError(f"Unsupported textual encoding: {config.get('type')}")
    return DAEPhonemeTokenizer(config["phoneme_map"], config.get("lexicon"))
