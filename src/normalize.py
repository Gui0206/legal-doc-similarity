"""Normalização de texto antes de gerar o fingerprint.

Objetivo: remover variação que não muda o *conteúdo* do documento
(ruído de OCR, cabeçalho/rodapé, numeração de página, espaçamento) para que
duas cópias com pequenas diferenças de formatação colidam no mesmo hash/shingles.
"""
from __future__ import annotations

import re
import unicodedata

# Padrões de "lixo" típico de documentos jurídicos digitalizados.
_PAGE_HEADER = re.compile(
    r"^\s*(f[óo]lha?s?\s+\d+|p[áa]g(?:ina)?\.?\s*\d+(?:\s*/\s*\d+)?|"
    r"tribunal de justi[çc]a.*|poder judici[áa]rio.*|n[ºo]\s*\d{4,}.*)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_PROTOCOL = re.compile(r"\b(?:doc|id|protocolo)\s*[:#]?\s*[0-9a-f\-]{6,}\b", re.IGNORECASE)
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"\w+", re.UNICODE)
_SENT_SPLIT = re.compile(r"[.;:!?\n]+")


def normalize(text: str) -> str:
    """Texto canônico usado para hashing e shingling."""
    text = unicodedata.normalize("NFC", text)
    text = _PAGE_HEADER.sub(" ", text)
    text = _PROTOCOL.sub(" ", text)
    text = text.lower()
    # remove pontuação isolada, mantém letras/números/acentos
    text = re.sub(r"[^\w\sáàâãéêíóôõúüç]", " ", text)
    text = _WS.sub(" ", text).strip()
    return text


def tokens(text: str) -> list[str]:
    """Tokens de palavra do texto já normalizado."""
    return _TOKEN.findall(text)


def sentences(text: str) -> list[str]:
    """Sentenças aproximadas (para o edit ratio / alinhamento)."""
    parts = [p.strip() for p in _SENT_SPLIT.split(text)]
    return [p for p in parts if p]
