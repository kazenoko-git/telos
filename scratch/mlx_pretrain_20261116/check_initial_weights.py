"""Capture actual public-launcher initial weights before dataset loading/updates."""
import argparse
import hashlib
import json
from pathlib import Path
import mlx.core as mx
from mlx.utils import tree_flatten
from telos.training.trainer_mlx import UnifiedMLXTrainer
from telos.train.cli import train

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
original=UnifiedMLXTrainer.__init__
class Captured(Exception):pass

def capture(self,*positional,**keywords):
    original(self,*positional,**keywords)
    path=args.output/'initial.safetensors'
    self.model.save_weights(str(path))
    report=dict(mlx_version=mx.__version__,mlx_module=mx.__file__,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                parameters=sum(value.size for _,value in tree_flatten(self.model.parameters())),
                model=self.cfg['model'],seed=self.seed)
    (args.output/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    raise Captured()

UnifiedMLXTrainer.__init__=capture
try:
    train(paradigm='ar',params='50M',tokens=64*16384,hardware='mlx',batch_size=2,grad_accum=8,
          seq_len=1024,data_path='/Users/ivansamuel/Documents/Projects/telos/data/english_corpus_5b.bin',
          synthetic=False,autotune=False,fused_step=True,async_eval=False,
          experimental_qkv_rope=True,experimental_packed_swiglu=True,force_fused_attention=True,
          compiled_accumulation=True,max_lr=4e-4,min_lr=4e-5,weight_decay=.01,
          checkpoint_dir=str(args.output/'unused'),save_every=1000,seed=42)
except Captured:pass
finally:UnifiedMLXTrainer.__init__=original
