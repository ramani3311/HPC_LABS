#include "utils.h"
#include <random>
#include <ctime>

void initialize_particles(std::vector<Particle>& particles) {
    std::mt19937 gen(std::time(0));
    std::uniform_real_distribution<double> dis(0.0, 1.0);

    for (auto &p : particles) {
        p.x = dis(gen);
        p.y = dis(gen);
    }
}