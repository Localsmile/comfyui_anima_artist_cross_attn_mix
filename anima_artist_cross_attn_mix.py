import re
import torch
from einops import rearrange


ARTIST_PATTERNS = [
    re.compile(r"@\(((?:[^()\\]|\\\(|\\\))+?):(-?\d+(?:\.\d+)?)\)"),
    re.compile(r"@\(((?:[^()\\]|\\\(|\\\))+?)\)"),
    re.compile(r"\(@((?:[^()\\]|\\\(|\\\))+?):(-?\d+(?:\.\d+)?)\)"),
    re.compile(r"@([\w\u4e00-\u9fff\uac00-\ud7af\-]+)"),
]


def parse_artists(text):
    spans = []
    consumed = [False] * len(text)
    for pattern in ARTIST_PATTERNS:
        for m in pattern.finditer(text):
            if any(consumed[m.start():m.end()]):
                continue
            groups = m.groups()
            name = groups[0].strip()
            weight = float(groups[1]) if len(groups) > 1 and groups[1] is not None else 1.0
            spans.append((m.start(), m.end(), name, weight))
            for i in range(m.start(), m.end()):
                consumed[i] = True
    spans.sort(key=lambda x: x[0])
    return spans


def cleanup_text(text):
    text = re.sub(r"(?:\s*,\s*){2,}", ", ", text)
    text = re.sub(r"^\s*,\s*", "", text)
    text = re.sub(r"\s*,\s*$", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def escape_for_native(name):
    out = []
    i = 0
    while i < len(name):
        ch = name[i]
        if ch == "\\" and i + 1 < len(name) and name[i + 1] in "()":
            out.append(name[i:i + 2])
            i += 2
            continue
        if ch in "()":
            out.append("\\" + ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def artist_token_str(name, weight=1.0):
    safe = escape_for_native(name)
    core = f"@{safe}"
    if weight == 1.0:
        return core
    return f"({core}:{weight:g})"


def single_artist_text(text, spans, keep_idx):
    parts = []
    last = 0
    for idx, (s, e, name, _) in enumerate(spans):
        parts.append(text[last:s])
        if idx == keep_idx:
            parts.append(artist_token_str(name, 1.0))
        last = e
    parts.append(text[last:])
    return cleanup_text("".join(parts))


def to_native_text(text, spans):
    if not spans:
        return text
    parts = []
    last = 0
    for s, e, name, w in spans:
        parts.append(text[last:s])
        parts.append(artist_token_str(name, w))
        last = e
    parts.append(text[last:])
    return "".join(parts)


class _MixedCrossAttn(torch.nn.Module):
    def __init__(self, orig_attn, kv_cache_per_artist, weights):
        super().__init__()
        self.orig = orig_attn
        self.kv_cache_per_artist = kv_cache_per_artist
        self.weights = weights

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        if transformer_options is None:
            transformer_options = {}
        cou = transformer_options.get("cond_or_uncond", None)

        q = self.orig.q_proj(x)
        q = rearrange(q, "b ... (h d) -> b ... h d",
                      h=self.orig.n_heads, d=self.orig.head_dim)
        q = self.orig.q_norm(q)

        B = q.shape[0]
        if cou is not None and len(cou) > 0 and B % len(cou) == 0:
            per_chunk = B // len(cou)
            chunks = []
            for i, cu in enumerate(cou):
                q_chunk = q[i * per_chunk:(i + 1) * per_chunk]
                if cu == 0:
                    chunks.append(self._mix(q_chunk, transformer_options))
                else:
                    ctx_chunk = context[i * per_chunk:(i + 1) * per_chunk]
                    chunks.append(self._standard(q_chunk, ctx_chunk, transformer_options))
            out = torch.cat(chunks, dim=0)
        else:
            out = self._mix(q, transformer_options)

        return self.orig.output_dropout(self.orig.output_proj(out))

    def _mix(self, q, transformer_options):
        per_chunk = q.shape[0]
        bare_k, bare_v = self.kv_cache_per_artist[0]
        if bare_k.shape[0] != per_chunk:
            bare_k = bare_k.expand(per_chunk, *bare_k.shape[1:])
            bare_v = bare_v.expand(per_chunk, *bare_v.shape[1:])
        bare_attn = self.orig.attn_op(
            q, bare_k, bare_v, transformer_options=transformer_options)

        sum_w = sum(self.weights)
        out = (1.0 - sum_w) * bare_attn
        for (k_i, v_i), w in zip(self.kv_cache_per_artist[1:], self.weights):
            if k_i.shape[0] != per_chunk:
                k_use = k_i.expand(per_chunk, *k_i.shape[1:])
                v_use = v_i.expand(per_chunk, *v_i.shape[1:])
            else:
                k_use, v_use = k_i, v_i
            artist_attn = self.orig.attn_op(
                q, k_use, v_use, transformer_options=transformer_options)
            out = out + w * artist_attn
        return out

    def _standard(self, q, context, transformer_options):
        k = self.orig.k_proj(context)
        v = self.orig.v_proj(context)
        k = rearrange(k, "b ... (h d) -> b ... h d",
                      h=self.orig.n_heads, d=self.orig.head_dim)
        v = rearrange(v, "b ... (h d) -> b ... h d",
                      h=self.orig.n_heads, d=self.orig.head_dim)
        k = self.orig.k_norm(k)
        v = self.orig.v_norm(v)
        return self.orig.attn_op(
            q, k, v, transformer_options=transformer_options)


def _build_kv_cache(blocks, adapter_outs):
    cache = []
    for block in blocks:
        attn = block.cross_attn
        per_block = []
        for adapter_out in adapter_outs:
            k = attn.k_proj(adapter_out)
            v = attn.v_proj(adapter_out)
            k = rearrange(k, "b ... (h d) -> b ... h d",
                          h=attn.n_heads, d=attn.head_dim)
            v = rearrange(v, "b ... (h d) -> b ... h d",
                          h=attn.n_heads, d=attn.head_dim)
            k = attn.k_norm(k)
            v = attn.v_norm(v)
            per_block.append((k, v))
        cache.append(per_block)
    return cache


class AnimaArtistCrossAttnMix:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "conditioning")
    FUNCTION = "process"
    CATEGORY = "conditioning/anima"

    def process(self, model, clip, text):
        spans = parse_artists(text)

        if not spans:
            tokens = clip.tokenize(to_native_text(text, spans))
            cond = clip.encode_from_tokens_scheduled(tokens)
            return (model, cond)

        bare_text = single_artist_text(text, spans, keep_idx=-1)
        bare_cond = clip.encode_from_tokens_scheduled(clip.tokenize(bare_text))[0]

        per_artist_conds = []
        weights = []
        for i, (_, _, _, w) in enumerate(spans):
            tokens = clip.tokenize(single_artist_text(text, spans, i))
            cond_list = clip.encode_from_tokens_scheduled(tokens)
            per_artist_conds.append(cond_list[0])
            weights.append(float(w))

        base_tensor, base_ctx = bare_cond
        base_ctx = dict(base_ctx)

        payload = [
            {
                "qwen3_hidden": tensor,
                "t5xxl_ids": ctx.get("t5xxl_ids", None),
                "t5xxl_weights": ctx.get("t5xxl_weights", None),
            }
            for tensor, ctx in ([bare_cond] + per_artist_conds)
        ]

        m_patched = model.clone()
        cache_state = {"kv_cache": None}

        def model_wrapper(apply_model, args):
            input_x = args["input"]
            timestep_ = args["timestep"]
            c = args["c"]

            diffusion_model = m_patched.model.diffusion_model
            device = input_x.device
            dtype = input_x.dtype

            if cache_state["kv_cache"] is None:
                adapter_outs = []
                for p in payload:
                    qwen = p["qwen3_hidden"].to(device=device, dtype=dtype)
                    t5_ids = p["t5xxl_ids"]
                    t5_w = p["t5xxl_weights"]
                    if t5_ids is not None:
                        t5_ids = t5_ids.to(device=device)
                        if t5_ids.dim() == 1:
                            t5_ids = t5_ids.unsqueeze(0)
                    if t5_w is not None:
                        t5_w = t5_w.to(device=device, dtype=dtype)
                        if t5_w.dim() == 1:
                            t5_w = t5_w.view(1, -1, 1)
                    adapter_outs.append(
                        diffusion_model.preprocess_text_embeds(qwen, t5_ids, t5_w))
                cache_state["kv_cache"] = _build_kv_cache(
                    diffusion_model.blocks, adapter_outs)

            kv_cache = cache_state["kv_cache"]
            blocks = diffusion_model.blocks

            originals = [block.cross_attn for block in blocks]
            for idx, block in enumerate(blocks):
                block.cross_attn = _MixedCrossAttn(
                    originals[idx], kv_cache[idx], weights)

            try:
                return apply_model(input_x, timestep_, **c)
            finally:
                for block, orig in zip(blocks, originals):
                    block.cross_attn = orig

        m_patched.set_model_unet_function_wrapper(model_wrapper)
        return (m_patched, [[base_tensor, base_ctx]])


NODE_CLASS_MAPPINGS = {"AnimaArtistCrossAttnMix": AnimaArtistCrossAttnMix}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaArtistCrossAttnMix": "Anima Artist Cross-Attn Mix"}