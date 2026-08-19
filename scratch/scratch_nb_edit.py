import json

with open("notebooks/Training_Suites.ipynb", "r") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = "".join(cell.get("source", []))
        if "# PIPELINE DEFINITION" in source:
            new_source = [
                "# PIPELINE DEFINITION\n",
                "start_time = time.time()\n",
                "\n",
                "print(\"\\n>>> Training 50M 1:35 (From Scratch) <<<\")\n",
                "run_training_step(\"configs/phase_b_50m_1to35_mlx.yaml\")\n",
                "\n",
                "total_elapsed = (time.time() - start_time) / 3600.0\n",
                "print(\"=\" * 85)\n",
                "print(f\"ALL 50M TARGET RUNS COMPLETED SUCCESSFULLY IN {total_elapsed:.2f} HOURS!\")\n",
                "print(\"=\" * 85)\n"
            ]
            cell["source"] = new_source
            cell["outputs"] = []

with open("notebooks/Training_Suites.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
