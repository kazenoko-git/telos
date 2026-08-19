import json

with open("notebooks/shared/Optimization_Test_Suite.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "telos.training.core" in src:
            print("Found telos.training.core import!")
        if "execute_mlx_training_step" in src:
            print("Found execute_mlx_training_step!")
