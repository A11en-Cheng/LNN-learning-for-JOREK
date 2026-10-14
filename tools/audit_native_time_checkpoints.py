#!/usr/bin/env python3
"""Restore/reject real checkpoints without performing optimizer updates."""
import argparse,json,sys,tempfile
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.data.native_training import prepare_training_data
from src.data.native_window_io import atomic_json
from tools.train_native_field_curriculum import make_trainer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--new',required=True,type=Path);p.add_argument('--baseline',required=True,type=Path)
    a=p.parse_args();cfg=json.loads((a.new/'experiment.json').read_text());torch.set_num_threads(1)
    with tempfile.TemporaryDirectory(prefix='native-time-checkpoints-') as folder:
        bundle=prepare_training_data(cfg['data'],Path(folder)/'data');rows=[]
        for seed in cfg['seeds']:
            trainer=make_trainer(bundle,cfg,seed)
            before={k:v.clone() for k,v in trainer.model.state_dict().items()}
            rng=trainer.generator.get_state().clone()
            try:trainer.load(a.baseline/f'seed-{seed}'/'final.pt')
            except ValueError as error:reason=str(error)
            else:raise AssertionError('old decoder checkpoint silently accepted')
            assert trainer.update_count==0 and torch.equal(trainer.generator.get_state(),rng)
            assert all(torch.equal(v,trainer.model.state_dict()[k]) for k,v in before.items())
            trainer.load(a.new/f'seed-{seed}'/'final.pt')
            assert trainer.update_count==60 and trainer.last_state is not None
            trainer.last_state.validate_for_model(trainer.model)
            rows.append(dict(seed=seed,old_rejected=reason,old_rejection_atomic=True,
                final_update=trainer.update_count,memory_shape=list(trainer.last_state.memory.shape),
                optimizer_updates_performed=0))
        atomic_json(a.new/'checkpoint-verification.json',dict(scope=cfg['scope'],rows=rows))


if __name__=='__main__':main()
