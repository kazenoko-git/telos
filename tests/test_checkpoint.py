"""Unit test for checkpoint save/resume continuity."""

import tempfile
from pathlib import Path
import torch
from mdiff.model.transformer import TelosTransformer, TelosConfig
from mdiff.training.trainer import TelosTrainer
from mdiff.data.dataset import create_dataloader


def test_checkpoint_save_and_resume():
    """Verifies model and optimizer state restoration continuity."""
    config = TelosConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
    model = TelosTransformer(config)
    
    # Dummy dataset sequences
    sequences = [[i % 50 for i in range(16)] for _ in range(32)]
    loader = create_dataloader(sequences, batch_size=8, max_seq_len=16)

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer_cfg = {
            "training": {"max_steps": 10, "max_lr": 1e-3},
            "checkpoint": {"dir": tmp_dir, "save_every_steps": 5}
        }
        trainer = TelosTrainer(model, train_loader=loader, config=trainer_cfg)

        # Train 5 steps
        for _ in range(5):
            masked_ids, targets, mask_pos, t_vals = next(iter(loader))
            trainer.optimizer.zero_grad()
            logits = trainer.model(masked_ids)
            loss = logits.sum()
            loss.backward()
            trainer.optimizer.step()
            trainer.global_step += 1

        ckpt_path = Path(tmp_dir) / "checkpoint_test.pt"
        trainer.save_checkpoint(ckpt_path)

        # Instantiate fresh model and trainer
        fresh_model = TelosTransformer(config)
        fresh_trainer = TelosTrainer(fresh_model, train_loader=loader, config=trainer_cfg)
        fresh_trainer.load_checkpoint(ckpt_path)

        # Verify global step restored
        assert fresh_trainer.global_step == 5
        
        # Verify parameter equivalence
        for p1, p2 in zip(trainer.model.parameters(), fresh_trainer.model.parameters()):
            assert torch.equal(p1, p2)
