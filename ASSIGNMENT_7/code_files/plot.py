import pandas as pd
import matplotlib.pyplot as plt

# Read CSV
df = pd.read_csv("results2.csv")

threads = df["threads"]
time = df["time"]

# Compute speedup
t1 = time.iloc[0]
speedup = t1 / time

# Compute efficiency
efficiency = speedup / threads

# -------------------------
# Plot 1: Execution Time
# -------------------------
plt.figure()
plt.plot(threads, time, marker='o')
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time vs Threads")
plt.grid()
plt.savefig("execution_time2.png")

# -------------------------
# Plot 2: Speedup
# -------------------------
plt.figure()
plt.plot(threads, speedup, marker='o', label="Actual Speedup")
plt.plot(threads, threads, linestyle='--', label="Ideal Speedup")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Threads")
plt.legend()
plt.grid()
plt.savefig("speedup2.png")

# -------------------------
# Plot 3: Efficiency
# -------------------------
plt.figure()
plt.plot(threads, efficiency, marker='o')
plt.xlabel("Number of Threads")
plt.ylabel("Efficiency")
plt.title("Efficiency vs Threads")
plt.grid()
plt.savefig("efficiency2.png")

print("Graphs generated:")
print(" - execution_time.png")
print(" - speedup.png")
print(" - efficiency.png")