import json

with open("notebooks/shared/Optimization_Test_Suite.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        if "return {" in "".join(cell["source"]):
            # Check if it's missing the closing brace
            src = cell["source"]
            if "active_mem_gb" in src[-1] and "}" not in src[-1]:
                # Append the closing brace
                src[-1] = src[-1].rstrip() + "\n"
                src.append("    }\n")
                cell["source"] = src

with open("notebooks/shared/Optimization_Test_Suite.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
