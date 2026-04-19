import matplotlib.pyplot as plt

threads = [1,2,4,8,16]
times = [0.000014,0.000064,0.000097,0.000154,0.000248]
speedup = [times[0]/t for t in times]

title_str = '10x10x1000'
file_prefix = 'plot_10x10x1000'
plt.figure(figsize=(6,4))
plt.plot(threads, times, marker='o', linewidth=2)
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (s)')
plt.title(f'Time vs Threads ({title_str})')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{file_prefix}_time_vs_threads.png', dpi=300)

plt.figure(figsize=(6,4))
plt.plot(threads, speedup, marker='o', linewidth=2)
plt.xlabel('Number of Threads')
plt.ylabel('Speedup')
plt.title(f'Speedup vs Threads ({title_str})')
plt.grid(True)
plt.tight_layout()
plt.savefig(f'{file_prefix}_speedup_vs_threads.png', dpi=300)

plt.show()
