import json

with open("notebooks/Training_Suites.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        new_source = []
        for line in cell["source"]:
            if "run_training_step(\"configs/phase_b_50m_1to35_mlx.yaml\")" in line:
                new_source.append(line.replace(
                    "run_training_step(\"configs/phase_b_50m_1to35_mlx.yaml\")",
                    "run_training_step(\"configs/phase_b_50m_1to35_mlx.yaml\", resume_from=\"checkpoints/phase_b_50m_1to35_mlx/checkpoint_step_5000.safetensors\", resume_step=5000)"
                ))
            else:
                new_source.append(line)
        cell["source"] = new_source

with open("notebooks/Training_Suites.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook patched 2 successfully.")
