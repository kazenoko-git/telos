import json

with open("notebooks/shared/Optimization_Test_Suite.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "is_first_step=False" in src:
            src = src.replace("is_first_step=False", "is_first_step=(step == 0)")
            cell["source"] = [line + "\n" for line in src.split("\n")][:-1]

with open("notebooks/shared/Optimization_Test_Suite.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
