import subprocess
import os

configs = [
    ("configs/phase_b_50m_1to15_mlx.yaml", "checkpoints/phase_b_50m_1to10_mlx"),
    ("configs/phase_b_50m_1to20_mlx.yaml", "checkpoints/phase_b_50m_1to15_mlx"),
    ("configs/phase_b_50m_1to25_mlx.yaml", "checkpoints/phase_b_50m_1to20_mlx"),
    ("configs/phase_b_50m_1to30_mlx.yaml", "checkpoints/phase_b_50m_1to25_mlx"),
    ("configs/phase_b_50m_1to35_mlx.yaml", "checkpoints/phase_b_50m_1to30_mlx"),
]

def main():
    print("Starting 50M Parameter Suite (1:15 to 1:35)...")
    for config, resume_from in configs:
        print(f"\n======================================")
        print(f"Training with config: {config}")
        print(f"Resuming from: {resume_from}")
        print(f"======================================")
        
        cmd = [
            "uv", "run", "scripts/train_mlx.py",
            "--config", config,
            "--resume_from", resume_from
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error during training {config}. Stopping suite.")
            break
            
if __name__ == "__main__":
    main()
