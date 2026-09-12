"""A plain encoder for decoder checkpoints that ship without a sentence-transformers config
(ITER, LRAT, a checkpoint trained here).

It does what Tevatron's `--pooling eos` encoding does, which is how those checkpoints were trained
and indexed: the checkpoint's own tokenizer, which appends its end-of-sequence token, left padding,
the hidden state at the last position, L2 normalisation. On BrowseComp-Plus documents it reproduces
ITER's published vectors to 0.997 cosine. The module-built sentence-transformers pipeline used
before tokenised the same checkpoints without the end token and reached 0.57: the trained model
then pooled an ordinary content token and ranked short junk documents first.

The object has the surface the retriever expects from an encoder: `encode`, `max_seq_length`,
`device`."""
from __future__ import annotations

from typing import Optional


class DecoderEncoder:
    def __init__(self, path: str, *, pooling: str = "last_token", normalize: bool = True,
                 max_seq_length: int = 512, device: Optional[str] = None, torch_dtype=None,
                 batch_size: int = 32):
        if pooling not in ("last_token", "mean", "cls"):
            raise ValueError(f"unknown pooling {pooling!r}; choose last_token, mean or cls")
        import torch                                    # retrieval extra; checked after the cheap validation
        from transformers import AutoModel, AutoTokenizer
        self.pooling = pooling
        self.normalize = normalize
        self.max_seq_length = int(max_seq_length)
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(path, trust_remote_code=True, torch_dtype=torch_dtype).to(dev).eval()
        self._device = torch.device(dev)

    @property
    def device(self):
        return self._device

    def _pool(self, hidden, mask):
        import torch
        if self.pooling == "last_token":
            return hidden[:, -1]                          # left padding: the last position is the end token
        if self.pooling == "cls":
            first = mask.float().argmax(dim=1)            # left padding: the first real token
            return hidden[torch.arange(hidden.size(0), device=hidden.device), first]
        m = mask.unsqueeze(-1).to(hidden.dtype)
        return (hidden * m).sum(1) / m.sum(1).clamp(min=1)

    def encode(self, texts, batch_size: Optional[int] = None, convert_to_tensor: bool = False,
               normalize_embeddings: bool = True, show_progress_bar: bool = False, **_ignored):
        import torch
        if isinstance(texts, str):
            texts = [texts]
        bs = int(batch_size or self.batch_size)
        chunks = []
        rng = range(0, len(texts), bs)
        if show_progress_bar:
            try:
                from tqdm import tqdm
                rng = tqdm(rng, desc="encode", unit="batch")
            except ImportError:
                pass
        for i in rng:
            batch = self.tokenizer(texts[i:i + bs], max_length=self.max_seq_length, truncation=True,
                                   padding=True, return_tensors="pt").to(self._device)
            with torch.no_grad():
                hidden = self.model(**batch).last_hidden_state
            vec = self._pool(hidden, batch["attention_mask"])
            if normalize_embeddings and self.normalize:
                vec = torch.nn.functional.normalize(vec.float(), dim=-1)
            chunks.append(vec.float().cpu())
        out = torch.cat(chunks) if chunks else torch.zeros((0, self.model.config.hidden_size))
        return out if convert_to_tensor else out.numpy()


__all__ = ["DecoderEncoder"]
