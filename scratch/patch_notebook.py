import json
import sys

with open("notebooks/Training_Suites.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "def run_training_step" in src:
            new_src = src.replace("def run_training_step(config_path, upscaled_source=None):", "def run_training_step(config_path, upscaled_source=None, resume_from=None, resume_step=0):")
            new_src = new_src.replace(
                "    trainer = TelosMLXTrainer(model, cfg)\n    trainer.train()\n",
                "    if resume_from:\n        print(f'  [Resume] Loading weights from {resume_from}')\n        model.load_weights(resume_from)\n    \n    trainer = TelosMLXTrainer(model, cfg)\n    trainer.train(resume_step=resume_step)\n"
            )
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in new_src.split("\n")][:-1]
        
        elif "PIPELINE DEFINITION" in src:
            new_src = src.replace(
                "run_training_step(\"configs/phase_b_50m_1to35_mlx.yaml\")",
                "run_training_step(\"configs/phase_b_50m_1to35_mlx.yaml\", resume_from=\"checkpoints/phase_b_50m_1to35_mlx/checkpoint_step_5000.safetensors\", resume_step=5000)"
            )
            cell["source"] = [line + "\n" if not line.endswith("\n") else line for line in new_src.split("\n")][:-1]

with open("notebooks/Training_Suites.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook patched successfully.")
