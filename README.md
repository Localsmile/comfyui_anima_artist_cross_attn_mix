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

Anima generally needs per-artist weights of **2.0+**.

## NegPiP

Non-artist tags with a negative weight are subtracted (NegPiP), with no extra
node or dependency required:

```
1girl, @(artist_a:2.0), (speech bubble:-1.1), (signature:-1.3)
```

## License

MIT
