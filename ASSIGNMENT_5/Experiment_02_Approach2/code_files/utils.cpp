#include "utils.h"
#include <omp.h>
#include <random>
#include <ctime>

void interpolation(const std::vector<Particle>& particles, std::vector<std::vector<double>>& mesh) {
    int N = particles.size();
    int NX = mesh.size();
    int NY = mesh[0].size();

    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int i_idx = (int)(particles[i].x * (NX - 1));
        int j_idx = (int)(particles[i].y * (NY - 1));
        
        if (i_idx >= 0 && i_idx < NX && j_idx >= 0 && j_idx < NY) {
            #pragma omp atomic
            mesh[i_idx][j_idx] += 1.0;
        }
    }
}

void mover_parallel(std::vector<Particle>& particles, int num_threads) {
    int N = particles.size();
    
    #pragma omp parallel num_threads(num_threads)
    {
        std::mt19937 gen(omp_get_thread_num() + std::time(0));
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        #pragma omp for
        for (int i = 0; i < N; ++i) {
            particles[i].x += 0.01; 
            particles[i].y += 0.01;

            if (particles[i].x < 0.0 || particles[i].x > 1.0 || 
                particles[i].y < 0.0 || particles[i].y > 1.0) {
                
                particles[i].x = dis(gen);
                particles[i].y = dis(gen);
            }
        }
    }
}