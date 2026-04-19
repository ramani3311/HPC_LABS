#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "init.h"
#include "utils.h"

// Global variables
int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <input_file>\n", argv[0]);
        return 1;
    }

    int thread_list[] = {1, 2, 4, 8, 16};
    int num_cases = 5;
    double times[5];

    // Open input file
    FILE *file = fopen(argv[1], "rb");
    if (!file) {
        printf("Error opening input file\n");
        exit(1);
    }

    // Read NX, NY, NUM_Points, Maxiter
    fread(&NX, sizeof(int), 1, file);
    fread(&NY, sizeof(int), 1, file);
    fread(&NUM_Points, sizeof(int), 1, file);
    fread(&Maxiter, sizeof(int), 1, file);

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    // Read all points
    double *all_points = (double*)malloc(Maxiter * NUM_Points * 2 * sizeof(double));
    if (!all_points) { printf("Memory allocation failed\n"); exit(1); }
    size_t points_read = fread(all_points, sizeof(double), Maxiter * NUM_Points * 2, file);
    if (points_read != (size_t)(Maxiter * NUM_Points * 2)) {
        printf("Error reading points\n"); exit(1);
    }
    fclose(file);

    for (int tcase = 0; tcase < num_cases; tcase++) {
        int threads = thread_list[tcase];
        omp_set_num_threads(threads);

        double *mesh_value = (double*)calloc(GRID_X * GRID_Y, sizeof(double));
        double total_time = 0.0;

        for (int iter = 0; iter < Maxiter; iter++) {
            memset(mesh_value, 0, GRID_X * GRID_Y * sizeof(double));

            Points *points = (Points*)malloc(NUM_Points * sizeof(Points));
            double *src = all_points + iter * NUM_Points * 2;
            for (int i = 0; i < NUM_Points; i++) {
                points[i].x = src[2*i];
                points[i].y = src[2*i + 1];
            }

            double start = omp_get_wtime();
            interpolation(mesh_value, points);
            double end = omp_get_wtime();
            total_time += (end - start);

            free(points);

            if (threads == 1 && iter == 0) save_mesh(mesh_value);
        }

        times[tcase] = total_time;
        printf("Threads = %d | Time = %lf sec\n", threads, total_time);
        free(mesh_value);
    }

    free(all_points);

    // Python plot script
    FILE *py = fopen("plot.py", "w");
    fprintf(py, "import matplotlib.pyplot as plt\n\n");
    fprintf(py, "threads = [1,2,4,8,16]\n");
    fprintf(py, "times = [");
    for (int i = 0; i < num_cases; i++) {
        fprintf(py, "%lf", times[i]);
        if (i != num_cases-1) fprintf(py, ",");
    }
    fprintf(py, "]\n");
    fprintf(py, "speedup = [times[0]/t for t in times]\n\n");

    // Title and file prefix include NX, NY, NUM_Points
    fprintf(py, "title_str = '%dx%dx%d'\n", NX, NY, NUM_Points);
    fprintf(py, "file_prefix = 'plot_%dx%dx%d'\n", NX, NY, NUM_Points);

    // Plot Time vs Threads
    fprintf(py, "plt.figure(figsize=(6,4))\n");
    fprintf(py, "plt.plot(threads, times, marker='o', linewidth=2)\n");
    fprintf(py, "plt.xlabel('Number of Threads')\n");
    fprintf(py, "plt.ylabel('Execution Time (s)')\n");
    fprintf(py, "plt.title(f'Time vs Threads ({title_str})')\n");
    fprintf(py, "plt.grid(True)\n");
    fprintf(py, "plt.tight_layout()\n");
    fprintf(py, "plt.savefig(f'{file_prefix}_time_vs_threads.png', dpi=300)\n\n");

    // Plot Speedup vs Threads
    fprintf(py, "plt.figure(figsize=(6,4))\n");
    fprintf(py, "plt.plot(threads, speedup, marker='o', linewidth=2)\n");
    fprintf(py, "plt.xlabel('Number of Threads')\n");
    fprintf(py, "plt.ylabel('Speedup')\n");
    fprintf(py, "plt.title(f'Speedup vs Threads ({title_str})')\n");
    fprintf(py, "plt.grid(True)\n");
    fprintf(py, "plt.tight_layout()\n");
    fprintf(py, "plt.savefig(f'{file_prefix}_speedup_vs_threads.png', dpi=300)\n\n");

    fprintf(py, "plt.show()\n");
    fclose(py);

    system("python3 plot.py");

    return 0;
}