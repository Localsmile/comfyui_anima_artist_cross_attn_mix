# Anima Artist Cross-Attn Mix

A ComfyUI node for the **Anima** model that blends multiple artist tags so their
styles mix evenly instead of one artist dominating the rest.

## Install

Clone into `ComfyUI/custom_nodes/`:

```
cd ComfyUI/custom_nodes
git clone https://github.com/Localsmile/comfyui_anima_artist_cross_attn_mix
```

Restart ComfyUI. The node appears under **conditioning/anima** as
**Anima Artist Cross-Attn Mix**.

## Usage

Wire `model` and `clip` in, then connect the `model` and `conditioning` outputs
to your sampler. Write artist tags with an `@` prefix in the text box:

```
masterpiece, 1girl, @(artist_a:0.8), @(artist_b:2.0), @artist_c
```

### Artist tag syntax

- `@artist_name` - weight 1.0
- `@(artist name:2.0)` - explicit weight (spaces allowed)
- `(@artist:1.5)` - alternative explicit-weight form
- `@(artist:-2)` - negative weight subtracts that style

### Mix settings

- `mix_mode`
  - `legacy` (default): the original mix. Each stacked artist adds its full
    weight (artists usually want 2.0+), so the push grows with every tag:
    six artists at 2.0 already break the image.
  - `normalized`: identical to legacy for a single artist, but the stacked
    total is divided by √(number of artists). Three artists look like legacy
    without its color fringing, and six or ten artists still produce a clean
    image. Weights mean the same as in legacy. Recommended when you stack
    artists.
- `strength`: style multiplier in `normalized` mode. `1.0` matches legacy for
  a single artist; lower it if a very large stack looks overcooked.
- `start_percent` / `end_percent`: the part of sampling where the artist mix
  applies. Steps outside it use the prompt without artist tags; for example,
  `start_percent = 0.2` lets the composition settle before the style comes in.

Workflows saved with older versions load with the defaults above and keep
the original behavior.

## NegPiP

Non-artist tags with a negative weight are subtracted (NegPiP), with no extra
node or dependency required:

```
1girl, @(artist_a:2.0), (speech bubble:-1.1), (signature:-1.3)
```

`negpip_mode` chooses how negative tags are applied:

- `legacy` (default): the negative tag's own tokens are flipped inside the
  prompt. This is the original behavior.
- `branch`: each negative tag is encoded on its own and its effect is
  subtracted from the result, scaled by the weight. The rest of the prompt
  never sees the negative concept, which avoids the tug-of-war that can cause
  odd results. Each negative tag costs one extra text encode and one extra
  attention pass per block. It also works when the prompt has no artist tags.

## Compatibility

Works with Anima ControlNet-LLLite, both ComfyUI's native **Apply Anima
LLLite** node and the sd-scripts custom node, in either order. Model-wrapper
nodes applied before this one keep working.

## License

MIT
