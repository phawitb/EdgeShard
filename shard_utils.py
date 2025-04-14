import torch.nn as nn

def split_model(model, cut_layer_idx):
    wte = model.transformer.wte
    wpe = model.transformer.wpe
    drop = model.transformer.drop

    layers = list(model.transformer.h.children())
    part1_blocks = layers[:cut_layer_idx]
    part2_blocks = layers[cut_layer_idx:]
    ln_f = model.transformer.ln_f
    lm_head = model.lm_head

    return (wte, wpe, drop, part1_blocks), (part2_blocks, ln_f), lm_head
