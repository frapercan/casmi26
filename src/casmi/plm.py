"""Frozen chemical language model (PLM) embeddings for molecules."""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_PLM = "DeepChem/ChemBERTa-77M-MTR"  # native RoBERTa, no remote code


class PLMEmbedder:
    def __init__(self, name: str = DEFAULT_PLM, device: str | None = None, max_len: int = 256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name)
        self.model.eval().to(self.device)
        if self.device == "cuda":
            self.model.half()
        self.max_len = max_len

    @torch.inference_mode()
    def embed(self, smiles: list[str], batch_size: int = 256) -> np.ndarray:
        """Mean-pooled last hidden state, L2-normalised, float16."""
        out = []
        for i in range(0, len(smiles), batch_size):
            b = self.tok(smiles[i:i + batch_size], padding=True, truncation=True,
                         max_length=self.max_len, return_tensors="pt").to(self.device)
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).to(h.dtype)
            e = (h * m).sum(1) / m.sum(1).clamp(min=1)
            e = torch.nn.functional.normalize(e.float(), dim=-1)
            out.append(e.half().cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, self.model.config.hidden_size), np.float16)
