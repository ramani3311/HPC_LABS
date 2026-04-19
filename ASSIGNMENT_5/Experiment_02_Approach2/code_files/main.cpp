#include <iostream>
#include <vector>
#include <fstream>
#include <omp.h>
#include "utils.h"

int main() {
    int NX = 1000, NY = 400;
    long long NUM_PARTICLES = 14000000;
    int MAX_ITER = 10;
    std::vector<int> thread_counts = {2, 4, 8, 16};

    std::ofstream outfile("results_parallel.csv");
    outfile << "Grid,Threads,Avg_Interp,Avg_Mover,Total_Execution\n";

    for (int num_threads : thread_counts) {
        std::vector<Particle> particles(NUM_PARTICLES);
        initialize_particles(particles); 

        double total_mover_time = 0;
        double total_interp_time = 0;

        for (int iter = 0; iter < MAX_ITER; ++iter) {
            std::vector<std::vector<double>> mesh(NX, std::vector<double>(NY, 0.0));

            double t1 = omp_get_wtime();
            interpolation(particles, mesh);
            total_interp_time += (omp_get_wtime() - t1);

            double t2 = omp_get_wtime();
            mover_parallel(particles, num_threads);
            total_mover_time += (omp_get_wtime() - t2);
        }

        double avg_interp = total_interp_time / MAX_ITER;
        double avg_mover = total_mover_time / MAX_ITER;
        
        outfile << NX << "x" << NY << "," << num_threads << "," 
                << avg_interp << "," << avg_mover << "," 
                << (avg_interp + avg_mover) << "\n";
        
        std::cout << "Successfully completed " << num_threads << " threads." << std::endl;
    }

    outfile.close();
    return 0;
}