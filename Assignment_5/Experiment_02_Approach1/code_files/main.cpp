#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <iostream>
#include <fstream>

#include "init.h"
#include "utils.h"

int GRID_X, GRID_Y, NX, NY;
int NUM_Points, Maxiter;
double dx, dy;

int main(int argc, char **argv) {
    NX = 250; 
    NY = 100;
    Maxiter = 10; 
    NUM_Points = 10000; 
    int thread_count = 1;

    GRID_X = NX + 1;
    GRID_Y = NY + 1;
    dx = 1.0 / NX;
    dy = 1.0 / NY;

    omp_set_num_threads(thread_count); 

    double *grid_mesh = (double *) calloc(GRID_X * GRID_Y, sizeof(double));
    Points *particle_list = (Points *) calloc(NUM_Points, sizeof(Points));

    initializepoints(particle_list);

    double total_interp_acc = 0;
    double total_mover_acc = 0;

    printf("Iteration\tInterp_Time\tMover_Time\tTotal_Time\n");
    for (int iter = 0; iter < Maxiter; iter++) {
        double start_time = omp_get_wtime();
        
        interpolation(grid_mesh, particle_list);
        double mid_time = omp_get_wtime();
        
        mover_parallel(particle_list, dx, dy);
        double end_time = omp_get_wtime();

        double interp_dur = mid_time - start_time;
        double mover_dur = end_time - mid_time;
        
        total_interp_acc += interp_dur;
        total_mover_acc += mover_dur;

        printf("%d\t\t%lf\t%lf\t%lf\n", iter + 1, interp_dur, mover_dur, interp_dur + mover_dur);
    }

    std::ofstream spreadsheet("results.csv", std::ios::app);
    
    std::ifstream check_file("results.csv");
    if (check_file.peek() == std::ifstream::traits_type::eof()) {
        spreadsheet << "Approach,Grid,Threads,Particles,Avg_Interp,Avg_Mover,Total_Execution\n";
    }
    check_file.close();

    spreadsheet << "Deferred," << NX << "x" << NY << "," 
                << thread_count << "," 
                << NUM_Points << "," 
                << total_interp_acc / Maxiter << "," 
                << total_mover_acc / Maxiter << "," 
                << (total_interp_acc + total_mover_acc) << "\n";
    
    spreadsheet.close();
    
    printf("\nData saved to results.csv\n");

    save_mesh(grid_mesh);

    free(grid_mesh);
    free(particle_list);
    return 0;
}