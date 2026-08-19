import json

with open("notebooks/shared/Optimization_Test_Suite.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "for _ in range(warmup_steps):" in src:
            # We want to replace the first `is_first_step=(step == 0)` with `False` 
            # and leave the second one (under `for step in range(bench_steps):`)
            # Alternatively, we can just replace `for _ in range(warmup_steps):` 
            # with `for step in range(warmup_steps):`
            src = src.replace("for _ in range(warmup_steps):", "for step in range(warmup_steps):")
            cell["source"] = [line + "\n" for line in src.split("\n")][:-1]

with open("notebooks/shared/Optimization_Test_Suite.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
