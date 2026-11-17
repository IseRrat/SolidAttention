"""Exercise --tokenize with a metadata-only synthetic GGUF and zero tensors."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def string(value):
    value=value.encode("utf-8")
    return struct.pack("<Q",len(value))+value


def vocabulary(path):
    tokens=["<unk>","<s>","</s>","▁","a","b"]+[f"<0x{i:02X}>" for i in range(256)]
    entries={"general.architecture":(8,string("llama")),"tokenizer.ggml.model":(8,string("llama")),
             "tokenizer.ggml.tokens":(9,struct.pack("<IQ",8,len(tokens))+b"".join(string(token) for token in tokens)),
             "tokenizer.ggml.scores":(9,struct.pack("<IQ",6,len(tokens))+struct.pack("<"+"f"*len(tokens),*([-1.0]*len(tokens)))),
             "tokenizer.ggml.token_type":(9,struct.pack("<IQ",5,len(tokens))+struct.pack("<"+"i"*len(tokens),*([2,3,3,1,1,1]+[6]*256))),
             "tokenizer.ggml.unknown_token_id":(4,struct.pack("<I",0)),"tokenizer.ggml.bos_token_id":(4,struct.pack("<I",1)),
             "tokenizer.ggml.eos_token_id":(4,struct.pack("<I",2)),"tokenizer.ggml.add_bos_token":(7,b"\x01")}
    data=b"GGUF"+struct.pack("<IQQ",3,0,len(entries))
    for key,(kind,value) in entries.items(): data+=string(key)+struct.pack("<I",kind)+value
    data+=b"\0"*((-len(data))%32)
    path.write_bytes(data)


with tempfile.TemporaryDirectory(prefix="solid-tokenizer-") as temporary:
    root=Path(temporary); model=root/"vocab.gguf"; vocabulary(model)
    prompt=root/"prompt.txt"; prompt.write_text("ab a",encoding="utf-8")
    config={"schema_version":1,"model":str(model),"model_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),
            "prompt":str(prompt),"output":str(root/"unused.json"),"context_tokens":64,"batch_tokens":8,"microbatch_tokens":4,
            "max_new_tokens":4,"threads":1,"gpu_layers":99,"flash_attention":True,"kv_on_gpu":True,
            "ignore_eos":False,"sampling":"greedy","kv_dtype":"f16"}
    path=root/"config.json"; path.write_text(json.dumps(config))
    def run():
        result=subprocess.run([sys.argv[1],"--tokenize",str(path)],capture_output=True,text=True,timeout=30)
        if result.returncode: raise ValueError(result.stderr)
        return json.loads(result.stdout)
    first,second=run(),run()
    assert first==second and first["token_ids"][0]==1 and len(first["token_ids"])>1
    assert first["source_sha256"]==hashlib.sha256(prompt.read_bytes()).hexdigest()
    assert first["model_parts_sha256"]==[config["model_sha256"]]
    assert not (root/"unused.json").exists()
    config["model_sha256"]="0"*64; path.write_text(json.dumps(config))
    bad=subprocess.run([sys.argv[1],"--tokenize",str(path)],capture_output=True,text=True,timeout=30)
    assert bad.returncode and "SHA256 mismatch" in bad.stderr
print("metadata-only vocabulary tokenization and provenance passed; no model tensors")
