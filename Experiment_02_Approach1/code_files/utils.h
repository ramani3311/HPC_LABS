#ifndef UTILS_H
#define UTILS_H
#include "init.h"

void interpolation(double *grid_mesh, Points *particle_list);
void mover_serial(Points *particle_list, double step_x, double step_y);
void mover_parallel(Points *particle_list, double step_x, double step_y);
void save_mesh(double *grid_mesh);

#endif