"""OpenCompass model registration; add the repository root to PYTHONPATH."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from opencompass.models.base import BaseModel
from opencompass.registry import MODELS
from accuracy_client import RunnerClient


@MODELS.register_module()
class SolidAttentionModel(BaseModel):
    def __init__(self, path, runner, output, variant="solid", max_seq_len=131583,
                 truncation="error", meta_template=None, tokenizer_only=False, block_tokens=32, budget_tokens=0, **kwargs):
        if kwargs: raise ValueError(f"unsupported SolidAttention model options: {sorted(kwargs)}")
        super().__init__(path=path,max_seq_len=max_seq_len,meta_template=meta_template,tokenizer_only=tokenizer_only)
        self.client = RunnerClient(path,runner,output,variant,max_seq_len,truncation,block_tokens,budget_tokens)

    def generate(self, inputs, max_out_len, **kwargs):
        # These defaults preserve the runner's deterministic greedy sampling.
        accepted = {"temperature":0,"do_sample":False,"min_out_len":None}
        for key,value in kwargs.items():
            if key not in accepted or value != accepted[key]: raise ValueError(f"unsupported generation option: {key}")
        return [self.client.generate(text,max_out_len)[0]["text"] for text in inputs]

    def get_token_len(self, prompt):
        return len(self.client.tokenize(prompt))

    def get_ppl(self, inputs, mask_length=None):
        if mask_length is not None and len(mask_length) != len(inputs): raise ValueError("PPL mask count differs from inputs")
        return np.asarray([self.client.score(text,None if mask_length is None else mask_length[i])[0] for i,text in enumerate(inputs)])
