# vLLM-FL Integration

The standalone repository contains the active kernel and its tests. The
service path also needs a small, explicit change in the pinned
`vllm-plugin-FL` checkout:

1. `iluvatar.yaml` selects `vendor:iluvatar` for attention.
2. `IluvatarBackend.attention_backend()` returns the Iluvatar attention
   implementation.
3. The implementation gates the repaired Split-KV path on both
   `ILUVATAR_USE_OPTIMIZED=1` and `ILUVATAR_SPLIT_KV=1`.
4. Unsupported features and pure decode remain on native vLLM attention.
5. Old untracked FA3/RoPE/Warp/Pingpong/TMA drafts are not part of the active
   vendor package.

The independently tested RoPE+KV-cache prototype is intentionally absent
from this patch. The pinned vLLM compilation configuration disables its
fusion pass on non-ROCm platforms; simply adding a vendor method would not
make the service use it. The Split-KV patch is also an experimental candidate,
not an instruction to deploy it after the repaired 8k latency regression.

Apply the reviewable patch from the root of a clean checkout at the pinned
`flagos-2026-s2` commit:

```bash
git apply ../minicpm-attention-optimization-iluvatar/integration/vllm-plugin-FL-split-kv.patch
```

The patch is intentionally separate from the standalone kernel commit so the
framework integration can be reviewed independently. Do not apply it to
9031 in place; use an isolated staging checkout or the dedicated diagnostic
service first.
