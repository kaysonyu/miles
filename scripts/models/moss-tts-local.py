"""mossLite v0.1.1 global Qwen3 backbone and untied local audio decoder."""


def model_args(**kwargs) -> str:
    return (
        "--swiglu --num-layers 36 --hidden-size 2560 --ffn-hidden-size 9728 "
        "--num-attention-heads 32 --group-query-attention --num-query-groups 8 "
        "--use-rotary-position-embeddings --disable-bias-linear --normalization RMSNorm "
        "--norm-epsilon 1e-6 --rotary-base 1000000 --vocab-size 151936 --kv-channels 128 "
        "--qk-layernorm --moss-local-num-attention-heads 32 --moss-local-ffn-hidden-size 9728 "
        "--moss-local-rotary-base 1000000 --moss-local-layernorm-epsilon 1e-6 "
    )
