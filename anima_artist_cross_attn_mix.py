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
        object.__setattr__(self, "orig", orig_attn)
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


def _expand_sign_mask(sign, adapter_out):
    if sign is None:
        return None
    T = adapter_out.shape[1]
    L = sign.shape[1]
    mask = adapter_out.new_ones((sign.shape[0], T, 1))
    n = min(L, T)
    mask[:, :n, :] = sign[:, :n, :].to(mask.dtype)
    if bool(torch.all(mask == 1)):
        return None
    return mask


def _build_kv_cache(blocks, adapter_outs, sign_masks=None):
    if sign_masks is None:
        sign_masks = [None] * len(adapter_outs)
    cache = []
    for block in blocks:
        attn = block.cross_attn
        per_block = []
        for adapter_out, sign in zip(adapter_outs, sign_masks):
            ctx_v = adapter_out if sign is None else adapter_out * sign
            k = attn.k_proj(adapter_out)
            v = attn.v_proj(ctx_v)
            k = rearrange(k, "b ... (h d) -> b ... h d",
                          h=attn.n_heads, d=attn.head_dim)
            v = rearrange(v, "b ... (h d) -> b ... h d",
                          h=attn.n_heads, d=attn.head_dim)
            k = attn.k_norm(k)
            v = attn.v_norm(v)
            per_block.append((k.detach(), v.detach()))
        cache.append(per_block)
    return cache


def _frozen_tensor(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu()
    return value


def _cache_key(diffusion_model, device, dtype):
    return (id(diffusion_model), device.type, device.index, dtype)


class _AnimaCrossAttnMixWrapper:
    def __init__(self, payload, weights):
        self.payload = tuple(payload)
        self.weights = tuple(float(w) for w in weights)
        self.kv_cache_by_key = {}

    def cleanup(self, **_kwargs):
        self.kv_cache_by_key.clear()

    def __call__(self, apply_model, args):
        input_x = args["input"]
        timestep_ = args["timestep"]
        c = args["c"]

        base_model = getattr(apply_model, "__self__", None)
        diffusion_model = getattr(base_model, "diffusion_model", None)
        if diffusion_model is None or not hasattr(diffusion_model, "blocks"):
            raise RuntimeError(
                "AnimaArtistCrossAttnMix expected a ComfyUI model with "
                "diffusion_model.blocks."
            )

        device = input_x.device
        dtype = input_x.dtype
        key = _cache_key(diffusion_model, device, dtype)
        kv_cache = self.kv_cache_by_key.get(key)
        if kv_cache is None:
            kv_cache = self._build_cache(diffusion_model, device, dtype)
            self.kv_cache_by_key[key] = kv_cache

        blocks = diffusion_model.blocks
        originals = [block.cross_attn for block in blocks]
        for idx, block in enumerate(blocks):
            block.cross_attn = _MixedCrossAttn(
                originals[idx], kv_cache[idx], self.weights)

        try:
            return apply_model(input_x, timestep_, **c)
        finally:
            for block, orig in zip(blocks, originals):
                block.cross_attn = orig

    def _build_cache(self, diffusion_model, device, dtype):
        adapter_outs = []
        sign_masks = []
        with torch.no_grad():
            for p in self.payload:
                qwen = p["qwen3_hidden"].to(device=device, dtype=dtype)
                t5_ids = p["t5xxl_ids"]
                t5_w = p["t5xxl_weights"]
                if t5_ids is not None:
                    t5_ids = t5_ids.to(device=device)
                    if t5_ids.dim() == 1:
                        t5_ids = t5_ids.unsqueeze(0)
                sign = None
                if t5_w is not None:
                    t5_w = t5_w.to(device=device, dtype=dtype)
                    if t5_w.dim() == 1:
                        t5_w = t5_w.view(1, -1, 1)
                    if t5_ids is not None:
                        sign = torch.sign(t5_w)
                        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                        t5_w = t5_w.abs()
                adapter_out = diffusion_model.preprocess_text_embeds(qwen, t5_ids, t5_w)
                adapter_outs.append(adapter_out.detach())
                sign_masks.append(_expand_sign_mask(sign, adapter_out))
            return _build_kv_cache(diffusion_model.blocks, adapter_outs, sign_masks)


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
    DESCRIPTION = (
        "Blend multiple artist tags for the Anima model so their styles mix "
        "evenly instead of one artist dominating the others.\n\n"
        "Tag artists in the text box with @name or @(name:2.0) (Anima usually "
        "wants weights of 2.0+). Each artist is encoded separately and the "
        "cross-attention outputs are blended by weight.\n\n"
        "Built-in NegPiP: any non-artist tag with a negative weight, e.g. "
        "(speech bubble:-1.1), is subtracted - no ComfyUI-ppm or extra node "
        "needed.\n\n"
        "Wire both outputs (model + conditioning) into your sampler."
    )

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
                "qwen3_hidden": _frozen_tensor(tensor),
                "t5xxl_ids": _frozen_tensor(ctx.get("t5xxl_ids", None)),
                "t5xxl_weights": _frozen_tensor(ctx.get("t5xxl_weights", None)),
            }
            for tensor, ctx in ([bare_cond] + per_artist_conds)
        ]

        m_patched = model.clone()
        model_wrapper = _AnimaCrossAttnMixWrapper(payload, weights)
        m_patched.set_model_unet_function_wrapper(model_wrapper)
        return (m_patched, [[base_tensor, base_ctx]])


NODE_CLASS_MAPPINGS = {"AnimaArtistCrossAttnMix": AnimaArtistCrossAttnMix}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaArtistCrossAttnMix": "Anima Artist Cross-Attn Mix"}
