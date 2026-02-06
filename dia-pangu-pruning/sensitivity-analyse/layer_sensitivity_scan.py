# layer_sensitivity_scan.py
import argparse
import torch
import pandas as pd

from load_model import load_model
from eval_metric import logprob_distance
from layer_mask import mask_attention_layer


PROMPTS = [
    # English
    "Explain the principle of Fourier Transform in simple terms.",
    "Summarize how attention works in transformers, in 5 bullet points.",
    "Write a short Python function to compute factorial recursively.",
    "What is the difference between precision and recall? Give an example.",
    "Explain what caching does in autoregressive decoding.",
    "Write an email to request a meeting with a colleague next week.",
    # Chinese
    "用通俗的语言解释一下傅里叶变换的基本思想。",
    "用三句话解释什么是机器学习中的过拟合。",
    "请给出一个简单的例子说明什么是梯度下降。",
    "把下面这句话翻译成英文：注意力机制可以帮助模型聚焦关键信息。",
    "写一个简短的段落，介绍一下变压器模型的优势。",
    "请用一步一步的方式说明如何煮一碗面。",
]


def run_once(model, tokenizer, prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model(**inputs)
    return outputs.logits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="layer_sensitivity.csv")
    ap.add_argument("--mode", type=str, default="v_only", choices=["v_only", "qv"],
                    help="v_only = mask V only (recommended). qv = mask Q and V.")
    ap.add_argument("--sanity_check", action="store_true",
                    help="Check that disabling mask restores ~0 distance for one layer.")
    args = ap.parse_args()

    model, tokenizer = load_model()
    num_layers = len(model.model.layers)
    print(f"Detected num_layers={num_layers}, prompts={len(PROMPTS)}, mode={args.mode}")

    # ---- Baseline logits (no patch) ----
    base_logits = {}
    for p in PROMPTS:
        base_logits[p] = run_once(model, tokenizer, p)

    # ---- Optional sanity check: mask then disable -> should be ~0 ----
    if args.sanity_check:
        test_layer = 0
        enable_mask, disable_mask, restore = mask_attention_layer(model, test_layer, mode=args.mode)

        # apply mask then disable it, compare to baseline
        enable_mask()
        _ = run_once(model, tokenizer, PROMPTS[0])  # run once under mask (not used)
        disable_mask()
        logits_restored = run_once(model, tokenizer, PROMPTS[0])

        restore()
        sc = logprob_distance(base_logits[PROMPTS[0]], logits_restored)
        print(f"[SanityCheck] layer={test_layer}, disable-mask distance={sc:.10f} (should be extremely small)")

    results = []

    for layer in range(num_layers):
        enable_mask, disable_mask, restore = mask_attention_layer(model, layer, mode=args.mode)

        enable_mask()  # mask entire attention layer

        scores = []
        for p in PROMPTS:
            masked_logits = run_once(model, tokenizer, p)
            score = logprob_distance(base_logits[p], masked_logits)
            scores.append(score)

        score_mean = float(torch.tensor(scores).mean().item())
        score_std = float(torch.tensor(scores).std(unbiased=False).item())

        results.append((layer, score_mean, score_std, len(PROMPTS)))
        print(f"Layer {layer:02d}, mean={score_mean:.6f}, std={score_std:.6f}")

        # cleanup
        disable_mask()
        restore()

    df = pd.DataFrame(results, columns=["layer", "score_mean", "score_std", "prompt_count"])
    df.to_csv(args.out, index=False)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()

