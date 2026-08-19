import json
import matplotlib.pyplot as plt

with open("notebooks/Training_Suites.ipynb") as f:
    nb = json.load(f)

steps, ces = [], []

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        outputs = cell.get("outputs", [])
        for out in outputs:
            if out.get("name") == "stdout":
                text = "".join(out.get("text", []))
                for line in text.split("\n"):
                    if "ELBO Loss:" in line and "CE:" in line:
                        parts = line.split("|")
                        step_part = parts[0].strip().split()[1] # "2150/3051"
                        step = int(step_part.split("/")[0])
                        ce_part = [p for p in parts if "CE:" in p][0]
                        ce = float(ce_part.split(":")[1].strip())
                        steps.append(step)
                        ces.append(ce)

plt.figure(figsize=(10, 6))
plt.plot(steps, ces, label="Cross Entropy (CE)")
plt.xlabel("Step")
plt.ylabel("CE Loss")
plt.title("50M 1:25 Upscaled Training Loss (3051 Steps)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig("/Users/ivansamuel/.gemini/antigravity-ide/brain/41b584d5-6756-4262-977b-2d3c9b81f8a1/media__loss_curve.png")
print(f"Saved plot with {len(steps)} points.")
