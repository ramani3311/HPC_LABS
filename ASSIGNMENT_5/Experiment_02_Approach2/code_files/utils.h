#ifndef UTILS_H
#define UTILS_H

#include <vector>

struct Particle {
    double x, y;
};

void initialize_particles(std::vector<Particle>& particles);
void interpolation(const std::vector<Particle>& particles, std::vector<std::vector<double>>& mesh);
void mover_parallel(std::vector<Particle>& particles, int num_threads);

#endif