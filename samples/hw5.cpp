#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace param {
const int n_steps = 200000;
const double dt = 60;  // time step in seconds
const double eps = 1e-3;  // soften parameter to avoid singularities
const double G = 6.674e-11;
double gravity_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}
const double planet_radius = 1e7;
const double missile_speed = 1e6;
double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

void read_input(const char* filename, int& n, int& planet, int& asteroid,
    std::vector<double>& qx, std::vector<double>& qy, std::vector<double>& qz,
    std::vector<double>& vx, std::vector<double>& vy, std::vector<double>& vz,
    std::vector<double>& m, std::vector<std::string>& type) {
    std::ifstream fin(filename);
    fin >> n >> planet >> asteroid;
    qx.resize(n);
    qy.resize(n);
    qz.resize(n);
    vx.resize(n);
    vy.resize(n);
    vz.resize(n);
    m.resize(n);
    type.resize(n);
    for (int i = 0; i < n; i++) {
        fin >> qx[i] >> qy[i] >> qz[i] >> vx[i] >> vy[i] >> vz[i] >> m[i] >> type[i];
    }
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific
         << std::setprecision(std::numeric_limits<double>::digits10 + 1) << min_dist
         << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

// oen step of the simulation
void run_step(int step, int n, std::vector<double>& qx, std::vector<double>& qy,
    std::vector<double>& qz, std::vector<double>& vx, std::vector<double>& vy,
    std::vector<double>& vz, const std::vector<double>& m,
    const std::vector<std::string>& type) {
    // compute accelerations between each pair of bodies
    std::vector<double> ax(n), ay(n), az(n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (j == i) continue;
            double mj = m[j];
            if (type[j] == "device")
                mj = param::gravity_device_mass(mj, step * param::dt);
            
            double dx = qx[j] - qx[i];
            double dy = qy[j] - qy[i];
            double dz = qz[j] - qz[i];
            double dist3 =
                pow(dx * dx + dy * dy + dz * dz + param::eps * param::eps, 1.5);
            ax[i] += param::G * mj * dx / dist3;
            ay[i] += param::G * mj * dy / dist3;
            az[i] += param::G * mj * dz / dist3;
        }
    }

    // update velocities
    for (int i = 0; i < n; i++) {
        vx[i] += ax[i] * param::dt;
        vy[i] += ay[i] * param::dt;
        vz[i] += az[i] * param::dt;
    }

    // update positions
    for (int i = 0; i < n; i++) {
        qx[i] += vx[i] * param::dt;
        qy[i] += vy[i] * param::dt;
        qz[i] += vz[i] * param::dt;
    }
}

int main(int argc, char** argv) {
    if (argc != 3)
        throw std::runtime_error("must supply 2 arguments");

    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<std::string> type;

    auto distance = [&](int i, int j) -> double {
        double dx = qx[i] - qx[j];
        double dy = qy[i] - qy[j];
        double dz = qz[i] - qz[j];
        return sqrt(dx * dx + dy * dy + dz * dz);
    };

    // Problem 1, calculate minimum distance between planet and asteroid
    std::vector<int> device_ids;
    double min_dist = std::numeric_limits<double>::infinity();
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);

    std::vector<double> qx_orig = qx, qy_orig = qy, qz_orig = qz;
    std::vector<double> vx_orig = vx, vy_orig = vy, vz_orig = vz;
    std::vector<double> m_orig = m;

    
    for (int i = 0; i < n; i++) {
        if (type[i] == "device") {
            m[i] = 0;
            device_ids.push_back(i);
        }
    }
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) 
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
    
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        min_dist = std::min(min_dist, distance(planet, asteroid));
    }

    // Problem 2, find first time step when asteroid hits planet
    int hit_time_step = -2;
    read_input(argv[1], n, planet, asteroid, qx, qy, qz, vx, vy, vz, m, type);
    for (int step = 0; step <= param::n_steps; step++) {
        if (step > 0) 
            run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);
    
        double dx = qx[planet] - qx[asteroid];
        double dy = qy[planet] - qy[asteroid];
        double dz = qz[planet] - qz[asteroid];
        if (dx * dx + dy * dy + dz * dz < param::planet_radius * param::planet_radius) {
            hit_time_step = step;
            break;
        }
    }

    // Problem 3, find device id and missile cost to prevent collision
    int best_device_id = -1;
    double min_missile_cost = std::numeric_limits<double>::infinity();
    if (hit_time_step == -2) {  // no collision detected
        write_output(argv[2], min_dist, hit_time_step, -1, 0);
        return 0;
    }

    for (int id: device_ids) {
        qx = qx_orig;
        qy = qy_orig;
        qz = qz_orig;
        vx = vx_orig;
        vy = vy_orig;
        vz = vz_orig;
        m = m_orig;

        bool destroyed = false;
        int destroy_step = -1;
        bool hit_planet = false;

        for (int step = 0; step <= param::n_steps; step++) {
            if (!destroyed) {
                double dist = distance(planet, id);
                double missle_travel_dist = param::missile_speed * (step+1) * param::dt;
                if (missle_travel_dist >= dist) {  // device destroyed
                    destroyed = true;
                    destroy_step = step;
                    m[id] = 0;
                }
            }

            if (step > 0) 
                run_step(step, n, qx, qy, qz, vx, vy, vz, m, type);

            double dist_planet_asteroid = distance(planet, asteroid);
            if (dist_planet_asteroid < param::planet_radius) {
                hit_planet = true;
                
                if (dist_planet_asteroid < min_missile_cost) {
                    hit_planet = true;
                    break;
                }
            }
        }

        if (!hit_planet) {  // successful prevention of collision
            double missile_cost = param::get_missile_cost((destroy_step + 1) * param::dt);
            if (missile_cost < min_missile_cost) {
                min_missile_cost = missile_cost;
                best_device_id = id;
            }
        }
    }

    if (best_device_id == -1)
        min_missile_cost = 0;
    
    write_output(argv[2], min_dist, hit_time_step, best_device_id, min_missile_cost);
}
