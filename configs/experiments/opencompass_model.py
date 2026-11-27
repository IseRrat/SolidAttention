# OpenCompass 模型配置入口，指定本仓库的模型包装器与推理程序参数。
"""Import this model list from a pinned OpenCompass experiment config."""
import os
from scripts.opencompass_model import SolidAttentionModel

models = [dict(
    type=SolidAttentionModel,
    abbr="solid-"+variant,
    path=os.environ["SOLID_MODEL_CONFIG"],
    runner=os.environ["SOLID_RUNNER"],
    output=os.path.join(os.environ["SOLID_ACCURACY_OUTPUT"],variant),
    variant=variant,
    max_seq_len=int(os.environ.get("SOLID_CONTEXT_TOKENS","131583")),
    truncation="error",
    batch_size=1,
    run_cfg=dict(num_gpus=1,num_procs=1),
) for variant in ("origin","quant","solid")]
