#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "utils.h"

// Parallel interpolation using atomic updates on a single global mesh
void interpolation(double *mesh_value, Points *points) {

    int size = GRID_X * GRID_Y;

    #pragma omp parallel
    {
        double *local_mesh = (double*)calloc(size, sizeof(double));

        #pragma omp for schedule(static)
        for (int k = 0; k < NUM_Points; k++) {

            double x = points[k].x;
            double y = points[k].y;

            int i = (int)(x / dx);
            int j = (int)(y / dy);

            if (i >= NX) i = NX - 1;
            if (j >= NY) j = NY - 1;

            double X = i * dx;
            double Y = j * dy;

            double lx = x - X;
            double ly = y - Y;

            double w1 = (dx - lx) * (dy - ly);
            double w2 = ly * (dx - lx);
            double w3 = lx * (dy - ly);
            double w4 = lx * ly;

            // ✅ Safe indexing
            int idx1 = j * GRID_X + i;
            local_mesh[idx1] += w1;

            if (i + 1 < GRID_X) {
                int idx2 = j * GRID_X + (i + 1);
                local_mesh[idx2] += w3;
            }

            if (j + 1 < GRID_Y) {
                int idx3 = (j + 1) * GRID_X + i;
                local_mesh[idx3] += w2;
            }

            if (i + 1 < GRID_X && j + 1 < GRID_Y) {
                int idx4 = (j + 1) * GRID_X + (i + 1);
                local_mesh[idx4] += w4;
            }
        }

        // Combine results (intentionally slower)
        #pragma omp critical
        {
            for (int idx = 0; idx < size; idx++) {
                mesh_value[idx] += local_mesh[idx];
            }
        }

        free(local_mesh);
    }
}
void save_mesh(double *mesh_value) {
    FILE *fd = fopen("Mesh.out", "w");
    if (!fd) {
        printf("Error creating Mesh.out\n");
        exit(1);
    }
    for (int i = 0; i < GRID_Y; i++) {
        for (int j = 0; j < GRID_X; j++) {
            fprintf(fd, "%lf ", mesh_value[i * GRID_X + j]);
        }
        fprintf(fd, "\n");
    }
    fclose(fd);
}