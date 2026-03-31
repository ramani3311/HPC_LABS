import pandas as pd
import matplotlib.pyplot as plt
import os

file_path = 'results.csv'
if not os.path.exists(file_path):
    print(f"Error: {file_path} not found. Please run your C++ simulation first.")
    exit()

df = pd.read_csv(file_path)

plt.figure(figsize=(10, 6))

for grid in df['Grid'].unique():
    grid_data = df[df['Grid'] == grid].sort_values(by='Particles')
    
    if not grid_data.empty:
        plt.loglog(grid_data['Particles'], grid_data['Total_Execution'], 
                   marker='o', linestyle='-', label=f'Grid {grid}')

plt.xlabel('Number of Particles (Log Scale)')
plt.ylabel('Total Execution Time (Seconds, Log Scale)')
plt.title('Experiment 01: Execution Time Scaling (Log-Log Plot)')
plt.grid(True, which="both", ls="-", alpha=0.5)
plt.legend()
plt.savefig('experiment_01_scaling.png')
print("Saved: experiment_01_scaling.png")
plt.show()

plt.figure(figsize=(10, 6))

speedup_df = df[df['Particles'] == 14000000].sort_values(by='Threads')

if not speedup_df.empty:
    t2_row = speedup_df[speedup_df['Threads'] == 2]
    
    if not t2_row.empty:
        t2_time = t2_row['Avg_Mover'].values[0]
        speedup_df['Speedup'] = t2_time / speedup_df['Avg_Mover']

        plt.plot(speedup_df['Threads'], speedup_df['Speedup'], 
                 marker='s', color='red', linewidth=2, label='Measured Speedup')
        
        ideal_threads = speedup_df['Threads'].values
        ideal_speedup = ideal_threads / 2.0
        plt.plot(ideal_threads, ideal_speedup, 
                 linestyle='--', color='gray', label='Ideal Scaling (Ref: 2 Threads)')

        plt.xlabel('Number of Threads')
        plt.ylabel('Speedup (T2 / Tn)')
        plt.title('Experiment 02: Mover Speedup vs Threads (14M Particles)')
        plt.xticks([2, 4, 8, 16])
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.savefig('experiment_02_speedup.png')