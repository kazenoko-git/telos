import matplotlib.pyplot as plt

steps, ces = [], []

with open("logs/overnight_suite.log", "r") as f:
    for line in f:
        if "ELBO Loss:" in line and "CE:" in line and "/9536" in line:
            parts = line.split("|")
            step_part = parts[0].strip().split()[1] # "100/9536"
            step = int(step_part.split("/")[0])
            ce_part = [p for p in parts if "CE:" in p][0]
            ce = float(ce_part.split(":")[1].strip())
            steps.append(step)
            ces.append(ce)

if steps:
    plt.figure(figsize=(10, 6))
    plt.plot(steps, ces, label="Cross Entropy (CE)")
    plt.xlabel("Step")
    plt.ylabel("CE Loss")
    plt.title("50M 1:25 Upscaled Training Loss (9536 Steps)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("/Users/ivansamuel/.gemini/antigravity-ide/brain/41b584d5-6756-4262-977b-2d3c9b81f8a1/media__loss_curve.png")
    print(f"Saved plot with {len(steps)} points.")
else:
    print("No lines found for 9536 steps.")
