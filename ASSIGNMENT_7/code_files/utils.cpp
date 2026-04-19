#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "utils.h"

double min_val, max_val;

// interpolation 
void interpolation(double *mesh_value, Points *points) {

    int total_size = GRID_X * GRID_Y;
    int nthreads = omp_get_max_threads();

    // thread-local meshes
    double **local_mesh = (double **)malloc(nthreads * sizeof(double *));
    for (int t = 0; t < nthreads; t++) {
        local_mesh[t] = (double *)calloc(total_size, sizeof(double));
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        double *lm = local_mesh[tid];

        #pragma omp for
        for (int p = 0; p < NUM_Points; p++) {

            if (points[p].is_void) continue;

            double x = points[p].x;
            double y = points[p].y;

            int i = (int)(x / dx);
            int j = (int)(y / dy);

            if (i < 0 || i >= GRID_X - 1 || j < 0 || j >= GRID_Y - 1) continue;

            double Xi = i * dx;
            double Yj = j * dy;

            double lx = x - Xi;
            double ly = y - Yj;

            // ✅ EXACT formula from lab (NO division)
            double w1 = (dx - lx) * (dy - ly);
            double w2 = ly * (dx - lx);
            double w3 = lx * (dy - ly);
            double w4 = lx * ly;

            int idx1 = j * GRID_X + i;
            int idx2 = j * GRID_X + (i + 1);
            int idx3 = (j + 1) * GRID_X + i;
            int idx4 = (j + 1) * GRID_X + (i + 1);

            // fi = 1 → just add weights
            lm[idx1] += w1;
            lm[idx2] += w2;
            lm[idx3] += w3;
            lm[idx4] += w4;
        }
    }

    // deterministic merge
    for (int t = 0; t < nthreads; t++) {
        for (int i = 0; i < total_size; i++) {
            mesh_value[i] += local_mesh[t][i];
        }
        free(local_mesh[t]);
    }
    free(local_mesh);
}

// normalization to [-1, 1]
void normalization(double *mesh_value) {

    min_val = 1e18;
    max_val = -1e18;

    #pragma omp parallel for reduction(min:min_val) reduction(max:max_val)
    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        if (mesh_value[i] < min_val) min_val = mesh_value[i];
        if (mesh_value[i] > max_val) max_val = mesh_value[i];
    }

    double range = max_val - min_val;
    if (range == 0.0) range = 1.0;

    #pragma omp parallel for
    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        mesh_value[i] = 2.0 * (mesh_value[i] - min_val) / range - 1.0;
    }
}

// mover via reverse-interpolation
void mover(double *mesh_value, Points *points) {

    #pragma omp parallel for
    for (int p = 0; p < NUM_Points; p++) {

        if (points[p].is_void) continue;

        double x = points[p].x;
        double y = points[p].y;

        int i = (int)(x / dx);
        int j = (int)(y / dy);

        if (i < 0 || i >= GRID_X - 1 || j < 0 || j >= GRID_Y - 1) {
            points[p].is_void = 1;
            continue;
        }

        double Xi = i * dx;
        double Yj = j * dy;

        double lx = x - Xi;
        double ly = y - Yj;

        // same weights
        double w1 = (dx - lx) * (dy - ly);
        double w2 = ly * (dx - lx);
        double w3 = lx * (dy - ly);
        double w4 = lx * ly;

        int idx1 = j * GRID_X + i;
        int idx2 = j * GRID_X + (i + 1);
        int idx3 = (j + 1) * GRID_X + i;
        int idx4 = (j + 1) * GRID_X + (i + 1);

        double Fi = w1 * mesh_value[idx1] +
                    w2 * mesh_value[idx2] +
                    w3 * mesh_value[idx3] +
                    w4 * mesh_value[idx4];

        points[p].x += Fi * dx;
        points[p].y += Fi * dy;

        if (points[p].x < 0 || points[p].x > 1 ||
            points[p].y < 0 || points[p].y > 1) {
            points[p].is_void = 1;
        }
    }
}

// denormalization
void denormalization(double *mesh_value) {

    double range = max_val - min_val;
    if (range == 0.0) range = 1.0;

    #pragma omp parallel for
    for (int i = 0; i < GRID_X * GRID_Y; i++) {
        mesh_value[i] = (mesh_value[i] + 1.0) * 0.5 * range + min_val;
    }
}

// count particles that went beyond the domain
long long int void_count(Points *points) {

    long long int voids = 0;

    #pragma omp parallel for reduction(+:voids)
    for (int i = 0; i < NUM_Points; i++) {
        voids += (int)points[i].is_void;
    }

    return voids;
}

// Write mesh to file
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