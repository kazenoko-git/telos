import tempfile
from pathlib import Path
import torch
from telos.models import TelosTransformer, TelosConfig
from telos.training import UnifiedPyTorchTrainer


def test_checkpoint_save_and_resume():
    """Verifies model and optimizer state restoration continuity."""
    config = TelosConfig(vocab_size=50, d_model=32, n_layers=2, n_heads=2, max_seq_len=16)
    model = TelosTransformer(config)

    trainer_cfg = {
        "model": {"vocab_size": 50, "seq_len": 16},
        "training": {"max_steps": 10, "max_lr": 1e-3, "batch_size": 4},
        "checkpoint": {"save_every_steps": 5}
    }

    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer_cfg["checkpoint"]["dir"] = tmp_dir
        trainer = UnifiedPyTorchTrainer("mdlm", model, trainer_cfg, device_type="cpu")

        # Simulate 5 steps of training
        for _ in range(5):
            dummy_batch = torch.randint(0, 50, (4, 16))
            trainer.optimizer.zero_grad()
            loss, _ = trainer._execute_microbatch(dummy_batch)
            loss.backward()
            trainer.optimizer.step()
            trainer.scheduler.step()
            trainer.global_step += 1

        ckpt_path = Path(tmp_dir) / "checkpoint_test.pt"
        trainer.save_checkpoint(ckpt_path)

        # Instantiate fresh model and trainer
        fresh_model = TelosTransformer(config)
        fresh_trainer = UnifiedPyTorchTrainer("mdlm", fresh_model, trainer_cfg, device_type="cpu")
        fresh_trainer.load_checkpoint(ckpt_path)

        # Verify global step restored
        assert fresh_trainer.global_step == 5
        
        # Verify parameter equivalence
        for p1, p2 in zip(trainer.model.parameters(), fresh_trainer.model.parameters()):
            assert torch.equal(p1, p2)
