// BY RAMANI
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <ctime>
#include <cmath>
#include <cstring>
#include "utils.h"

void interpolation(double *grid_mesh, Points *particle_list) {
    memset(grid_mesh, 0, (NX + 1) * (NY + 1) * sizeof(double));
    
    #pragma omp parallel for
    for (int i = 0; i < NUM_Points; i++) {
        int col = (int)(particle_list[i].x / dx);
        int row = (int)(particle_list[i].y / dy);
        
        if (col >= NX) col = NX - 1;
        if (row >= NY) row = NY - 1;

        double x_dist = (particle_list[i].x / dx) - col;
        double y_dist = (particle_list[i].y / dy) - row;

        grid_mesh[row * (NX + 1) + col] += (1.0 - x_dist) * (1.0 - y_dist);
        grid_mesh[row * (NX + 1) + (col + 1)] += x_dist * (1.0 - y_dist);
        grid_mesh[(row + 1) * (NX + 1) + col] += (1.0 - x_dist) * y_dist;
        grid_mesh[(row + 1) * (NX + 1) + (col + 1)] += x_dist * y_dist;
    }
}

void mover_parallel(Points *particle_list, double step_x, double step_y) {
    int deleted_count = 0;

    #pragma omp parallel
    {
        unsigned int seed = (unsigned int)time(NULL) ^ omp_get_thread_num();
        #pragma omp for reduction(+:deleted_count)
        for (int i = 0; i < NUM_Points; i++) {
            double drift = sin(particle_list[i].x) * 0.00001; 
            particle_list[i].x += (double)rand_r(&seed) / RAND_MAX * 0.02 - 0.01 + drift;
            particle_list[i].y += (double)rand_r(&seed) / RAND_MAX * 0.02 - 0.01;

            if (particle_list[i].x < 0 || particle_list[i].x > 1.0 || 
                particle_list[i].y < 0 || particle_list[i].y > 1.0) {
                particle_list[i].x = -1.0;
                deleted_count++;
            }
        }
    }

    #pragma omp parallel
    {
        unsigned int seed = (unsigned int)time(NULL) ^ (omp_get_thread_num() + 10);
        #pragma omp for
        for (int i = 0; i < NUM_Points; i++) {
            if (particle_list[i].x < 0) {
                particle_list[i].x = (double)rand_r(&seed) / RAND_MAX;
                particle_list[i].y = (double)rand_r(&seed) / RAND_MAX;
            }
        }
    }
}

void save_mesh(double *grid_mesh) {
    FILE *fp = fopen("mesh_output.csv", "w");
    if (!fp) return;
    for (int i = 0; i <= NY; i++) {
        for (int j = 0; j <= NX; j++) {
            fprintf(fp, "%lf%s", grid_mesh[i * (NX + 1) + j], (j == NX ? "" : ","));
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    printf("Simulation data exported successfully.\n");
}