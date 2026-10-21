import math
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


NEGPIP_PATTERN = re.compile(r"\(((?:[^()\\]|\\\(|\\\))+?):\s*(-\d+(?:\.\d+)?)\s*\)")


def parse_negpip(text, artist_spans):
    taken = [False] * len(text)
    for s, e, _, _ in artist_spans:
        for i in range(s, e):
            taken[i] = True
    spans = []
    for m in NEGPIP_PATTERN.finditer(text):
        if any(taken[m.start():m.end()]):
            continue
        spans.append((m.start(), m.end(), m.group(1).strip(), float(m.group(2))))
    return spans


def branch_text(text, artist_spans, keep_artist=-1, neg_spans=(), keep_neg=-1):
    """Drop every artist tag except keep_artist (written at weight 1.0) and
    every negative tag except keep_neg (written as a plain positive tag)."""
    edits = [(s, e, artist_token_str(name, 1.0) if idx == keep_artist else "")
             for idx, (s, e, name, _) in enumerate(artist_spans)]
    edits += [(s, e, inner if idx == keep_neg else "")
              for idx, (s, e, inner, _) in enumerate(neg_spans)]
    edits.sort(key=lambda x: x[0])
    parts = []
    last = 0
    for s, e, repl in edits:
        parts.append(text[last:s])
        parts.append(repl)
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


MIX_MODES = ["legacy", "normalized"]
NEGPIP_MODES = ["legacy", "branch"]

# Per-token cap on |output| / |output before subtraction| (L1), as in
# Normalized Attention Guidance. Applied only after subtracting negative
# artists or branch NegPiP tags, so they can't push tokens far outside the
# range the model was trained on.
NORM_GUARD_TAU = 2.0
_EPS = 1e-6


def _l1(x):
    return x.abs().mean(dim=-1, keepdim=True)


def _norm_guard(z, ref, tau=NORM_GUARD_TAU):
    ratio = _l1(z) / (_l1(ref) + _EPS)
    return z * torch.clamp(tau / (ratio + _EPS), max=1.0)


class _MixedCrossAttn(torch.nn.Module):
    # kv_cache_per_artist: [bare, artist_1..artist_n, neg_1..neg_m], where
    # neg_j is the bare prompt plus negative tag j written as a positive tag.
    def __init__(self, orig_attn, kv_cache_per_artist, weights, neg_weights=(),
                 mode="legacy", strength=1.0, artists_active=True):
        super().__init__()
        object.__setattr__(self, "orig", orig_attn)
        self.kv_cache_per_artist = kv_cache_per_artist
        self.weights = weights
        self.neg_weights = neg_weights
        self.mode = mode
        self.strength = strength
        self.artists_active = artists_active

    def forward(self, x, context=None, rope_emb=None, transformer_options=None):
        if transformer_options is None:
            transformer_options = {}
        cou = transformer_options.get("cond_or_uncond", None)

        # Run attn2 patches like the stock cross-attn does (e.g. native Anima
        # LLLite edits the query input here). Patched k/v only reach the
        # uncond path; the mixed path uses its cached per-branch K/V.
        k_in = v_in = context
        attn2_patches = transformer_options.get("patches", {}).get("attn2_patch", [])
        if attn2_patches:
            extra_options = transformer_options.copy()
            extra_options["n_heads"] = self.orig.n_heads
            extra_options["dim_head"] = self.orig.head_dim
            for patch in attn2_patches:
                out = patch(x, k_in, v_in, pe=rope_emb, attn_mask=None,
                            extra_options=extra_options)
                x = out.get("q", x)
                k_in = out.get("k", k_in)
                v_in = out.get("v", v_in)

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
                    sl = slice(i * per_chunk, (i + 1) * per_chunk)
                    chunks.append(self._standard(q_chunk, k_in[sl], v_in[sl],
                                                 transformer_options))
            out = torch.cat(chunks, dim=0)
        else:
            out = self._mix(q, transformer_options)

        return self.orig.output_dropout(self.orig.output_proj(out))

    def _attn(self, q, kv, transformer_options):
        k, v = kv
        if k.dtype != q.dtype:
            k, v = k.to(q.dtype), v.to(q.dtype)
        per_chunk = q.shape[0]
        if k.shape[0] != per_chunk:
            k = k.expand(per_chunk, *k.shape[1:])
            v = v.expand(per_chunk, *v.shape[1:])
        preferred = getattr(self.orig, "comfy_attention", None)
        if preferred is not None:
            return self.orig.attn_op(q, k, v, transformer_options=transformer_options,
                                     preferred_attention=preferred)
        return self.orig.attn_op(q, k, v, transformer_options=transformer_options)

    def _mix(self, q, transformer_options):
        bare = self._attn(q, self.kv_cache_per_artist[0], transformer_options)
        if not self.artists_active or not self.weights:
            out = bare
        elif self.mode == "legacy":
            out = self._mix_legacy(q, bare, transformer_options)
        else:
            out = self._mix_normalized(q, bare, transformer_options)
        if self.neg_weights:
            out = self._subtract_negpip(q, bare, out, transformer_options)
        return out.to(q.dtype)

    def _mix_legacy(self, q, bare_attn, transformer_options):
        sum_w = sum(self.weights)
        out = (1.0 - sum_w) * bare_attn
        for kv, w in zip(self.kv_cache_per_artist[1:], self.weights):
            out = out + w * self._attn(q, kv, transformer_options)
        return out

    def _subtract_negpip(self, q, bare, mixed, transformer_options):
        # Branch NegPiP: each negative tag is encoded as a positive tag on the
        # bare prompt, and its delta from the bare prompt is subtracted from
        # the mixed output, scaled by |weight|. Unlike flipping the value sign
        # inside one prompt, the concept stays out of every other branch.
        bare = bare.float()
        mixed = mixed.float()
        z = mixed
        neg_kvs = self.kv_cache_per_artist[1 + len(self.weights):]
        for kv, w_abs in zip(neg_kvs, self.neg_weights):
            n = self._attn(q, kv, transformer_options).float()
            z = z - w_abs * (n - bare)
        return _norm_guard(z, mixed)

    def _mix_normalized(self, q, bare, transformer_options):
        # Legacy adds w_i * (artist_i - bare) for every artist, so N stacked
        # artists push the output N times as far and the image falls apart.
        # Here the sum is divided by sqrt(N): one artist matches legacy
        # exactly, and stacking grows only as sqrt(N). Negative artists are
        # subtracted the same way, after removing the part of their delta that
        # points along the positive style (the shared "has an artist tag"
        # direction), so they don't wash out the positive mix.
        bare = bare.float()
        d_pos = torch.zeros_like(bare)
        n_pos = 0
        negatives = []
        for kv, w in zip(self.kv_cache_per_artist[1:], self.weights):
            if w == 0:
                continue
            d = self._attn(q, kv, transformer_options).float() - bare
            if w > 0:
                d_pos = d_pos + w * d
                n_pos += 1
            else:
                negatives.append((d, -w))

        if n_pos:
            d_pos = d_pos * (self.strength / math.sqrt(n_pos))
        z = bare + d_pos
        if not negatives:
            return z

        heads = (*bare.shape[:-1], self.orig.n_heads, -1)
        d_pos_h = d_pos.view(heads)
        d_pos_sq = (d_pos_h * d_pos_h).sum(dim=-1, keepdim=True) + _EPS
        neg_scale = self.strength / math.sqrt(len(negatives))
        z_neg = z
        for d, w_abs in negatives:
            d = d.view(heads)
            coef = ((d * d_pos_h).sum(dim=-1, keepdim=True) / d_pos_sq).clamp(min=0)
            d = (d - coef * d_pos_h).reshape(bare.shape)
            z_neg = z_neg - neg_scale * w_abs * d
        return _norm_guard(z_neg, z)

    def _standard(self, q, k_in, v_in, transformer_options):
        k = self.orig.k_proj(k_in)
        v = self.orig.v_proj(v_in)
        k = rearrange(k, "b ... (h d) -> b ... h d",
                      h=self.orig.n_heads, d=self.orig.head_dim)
        v = rearrange(v, "b ... (h d) -> b ... h d",
                      h=self.orig.n_heads, d=self.orig.head_dim)
        k = self.orig.k_norm(k)
        v = self.orig.v_norm(v)
        return self._attn(q, (k, v), transformer_options)


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


def _compute_dtype(base_model, fallback):
    for name in ("get_dtype_inference", "get_dtype"):
        fn = getattr(base_model, name, None)
        if callable(fn):
            return fn()
    return fallback


class _AnimaCrossAttnMixWrapper:
    def __init__(self, payload, weights, neg_weights=(), mode="legacy",
                 strength=1.0, sigma_range=None, inner_wrapper=None):
        self.payload = tuple(payload)
        # model_function_wrapper is a single slot; keep whatever was there
        # (e.g. the sd-scripts Anima LLLite node) and call it inside ours.
        self.inner_wrapper = inner_wrapper
        self.weights = tuple(float(w) for w in weights)
        self.neg_weights = tuple(float(w) for w in neg_weights)
        self.mode = mode
        self.strength = float(strength)
        self.sigma_range = sigma_range
        self.kv_cache_by_key = {}

    def _in_range(self, timestep):
        if self.sigma_range is None:
            return True
        start_sigma, end_sigma = self.sigma_range
        sigma = float(timestep.max()) if torch.is_tensor(timestep) else float(timestep)
        return end_sigma <= sigma <= start_sigma

    def cleanup(self, **kwargs):
        self.kv_cache_by_key.clear()
        inner_cleanup = getattr(self.inner_wrapper, "cleanup", None)
        if callable(inner_cleanup):
            inner_cleanup(**kwargs)

    def _call_next(self, apply_model, input_x, timestep_, c):
        if self.inner_wrapper is not None:
            return self.inner_wrapper(
                apply_model, {"input": input_x, "timestep": timestep_, "c": c})
        return apply_model(input_x, timestep_, **c)

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

        # Outside the active range only the bare prompt is used (no artists);
        # NegPiP still applies.
        artists_active = self._in_range(timestep_)

        device = input_x.device
        dtype = _compute_dtype(base_model, input_x.dtype)
        key = _cache_key(diffusion_model, device, dtype)
        kv_cache = self.kv_cache_by_key.get(key)
        if kv_cache is None:
            kv_cache = self._build_cache(diffusion_model, device, dtype)
            self.kv_cache_by_key[key] = kv_cache

        blocks = diffusion_model.blocks
        originals = [block.cross_attn for block in blocks]
        for idx, block in enumerate(blocks):
            block.cross_attn = _MixedCrossAttn(
                originals[idx], kv_cache[idx], self.weights,
                neg_weights=self.neg_weights, mode=self.mode,
                strength=self.strength, artists_active=artists_active)

        try:
            return self._call_next(apply_model, input_x, timestep_, c)
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
                "mix_mode": (MIX_MODES, {
                    "default": "legacy",
                    "tooltip": "legacy: the original mix; every stacked artist adds "
                               "its full weight, so many artists break the image. "
                               "normalized: same as legacy for one artist, but "
                               "stacking N artists grows as sqrt(N) instead of N.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Artist style multiplier in normalized mode. 1.0 "
                               "matches legacy for a single artist.",
                }),
                "start_percent": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Step fraction where the artist mix starts. Earlier "
                               "steps use the prompt without artist tags.",
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Step fraction where the artist mix ends. Later "
                               "steps use the prompt without artist tags.",
                }),
                "negpip_mode": (NEGPIP_MODES, {
                    "default": "legacy",
                    "tooltip": "legacy: negative tags flip their own token values "
                               "inside the prompt (original behavior). branch: each "
                               "negative tag is encoded separately and its effect "
                               "is subtracted, so it doesn't leak into the rest of "
                               "the prompt. Costs one extra encode per negative tag.",
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING")
    RETURN_NAMES = ("model", "conditioning")
    FUNCTION = "process"
    CATEGORY = "conditioning/anima"
    DESCRIPTION = (
        "Blend multiple artist tags for the Anima model so their styles mix "
        "evenly instead of one artist dominating the others.\n\n"
        "Tag artists in the text box with @name or @(name:2.0). Each artist is "
        "encoded separately and the cross-attention outputs are blended.\n\n"
        "Artists usually want weights of 2.0+. legacy mode (default): the "
        "original mix, where each stacked artist adds its full weight, so "
        "many artists break the image. normalized mode: identical to legacy "
        "for one artist, but the stacked total is divided by sqrt(number of "
        "artists), so heavy stacking keeps the legacy look without breaking. "
        "strength multiplies the style.\n\n"
        "Built-in NegPiP: any non-artist tag with a negative weight, e.g. "
        "(speech bubble:-1.1), is subtracted - no ComfyUI-ppm or extra node "
        "needed. negpip_mode=branch encodes each negative tag separately and "
        "subtracts its effect, which keeps it from leaking into the rest of "
        "the prompt.\n\n"
        "Wire both outputs (model + conditioning) into your sampler."
    )

    def process(self, model, clip, text, mix_mode="legacy", strength=1.0,
                start_percent=0.0, end_percent=1.0, negpip_mode="legacy"):
        spans = parse_artists(text)
        neg_spans = parse_negpip(text, spans) if negpip_mode == "branch" else []

        if not spans and not neg_spans:
            tokens = clip.tokenize(to_native_text(text, spans))
            cond = clip.encode_from_tokens_scheduled(tokens)
            return (model, cond)

        def encode(t):
            return clip.encode_from_tokens_scheduled(clip.tokenize(t))[0]

        bare_cond = encode(branch_text(text, spans, -1, neg_spans))

        per_artist_conds = []
        weights = []
        for i, (_, _, _, w) in enumerate(spans):
            per_artist_conds.append(encode(branch_text(text, spans, i, neg_spans)))
            weights.append(float(w))

        neg_conds = []
        neg_weights = []
        for j, (_, _, _, w) in enumerate(neg_spans):
            neg_conds.append(encode(branch_text(text, spans, -1, neg_spans, keep_neg=j)))
            neg_weights.append(abs(float(w)))

        base_tensor, base_ctx = bare_cond
        base_ctx = dict(base_ctx)

        payload = [
            {
                "qwen3_hidden": _frozen_tensor(tensor),
                "t5xxl_ids": _frozen_tensor(ctx.get("t5xxl_ids", None)),
                "t5xxl_weights": _frozen_tensor(ctx.get("t5xxl_weights", None)),
            }
            for tensor, ctx in ([bare_cond] + per_artist_conds + neg_conds)
        ]

        sigma_range = None
        if start_percent > 0.0 or end_percent < 1.0:
            model_sampling = model.get_model_object("model_sampling")
            sigma_range = (
                float(model_sampling.percent_to_sigma(start_percent)),
                float(model_sampling.percent_to_sigma(end_percent)),
            )

        m_patched = model.clone()
        model_wrapper = _AnimaCrossAttnMixWrapper(
            payload, weights, neg_weights=neg_weights, mode=mix_mode,
            strength=strength, sigma_range=sigma_range,
            inner_wrapper=m_patched.model_options.get("model_function_wrapper"))
        m_patched.set_model_unet_function_wrapper(model_wrapper)
        return (m_patched, [[base_tensor, base_ctx]])


NODE_CLASS_MAPPINGS = {"AnimaArtistCrossAttnMix": AnimaArtistCrossAttnMix}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaArtistCrossAttnMix": "Anima Artist Cross-Attn Mix"}
