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
    const double eps2 = eps * eps;
    const double G = 6.674e-11;
    double gravity_device_mass(double m0, double t) {
        return m0 + 0.5 * m0 * fabs(sin(t / 6000));
    }
    const double planet_radius = 1e7;
    const double missile_speed = 1e6;
    double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}  // namespace param

struct ParticleSystem {
    int n;
    int planet;
    int asteroid;

    std::vector<double> qx, qy, qz;
    std::vector<double> vx, vy, vz;
    std::vector<double> m;

    std::vector<bool> is_device;
    std::vector<int> device_ids;

    void resize(int size) {
        n = size;
        qx.resize(size); qy.resize(size); qz.resize(size);
        vx.resize(size); vy.resize(size); vz.resize(size);
        m.resize(size);
        is_device.resize(size, false);
    }

    void reset_to(const ParticleSystem& init) {
        qx = init.qx; qy = init.qy; qz = init.qz;
        vx = init.vx; vy = init.vy; vz = init.vz;
        m = init.m;
    }

    double disSquared(int i, int j) const {
        double dx = qx[i] - qx[j];
        double dy = qy[i] - qy[j];
        double dz = qz[i] - qz[j];
        return dx * dx + dy * dy + dz * dz;
    }

    double distance(int i, int j) const {
        return std::sqrt(disSquared(i, j));
    }
};

inline double get_device_mass(double m0, double t) {
    return m0 + 0.5 * m0 * fabs(sin(t / 6000));
}

ParticleSystem read_input(const char* filename) {
    std::ifstream fin(filename);

    ParticleSystem s;
    int n, planet, asteroid;
    fin >> n >> planet >> asteroid;

    s.resize(n);
    s.planet = planet;
    s.asteroid = asteroid;

    for (int i = 0; i < n; i++) {
        std::string type;
        fin >> s.qx[i] >> s.qy[i] >> s.qz[i]
            >> s.vx[i] >> s.vy[i] >> s.vz[i]
            >> s.m[i] >> type;

        if (type == "device") {
            s.is_device[i] = true;
            s.device_ids.push_back(i);
        } else {
            s.is_device[i] = false;
        }
    }
    return s;
}

void write_output(const char* filename, double min_dist, int hit_time_step,
    int gravity_device_id, double missile_cost) {
    std::ofstream fout(filename);
    fout << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1) 
         << min_dist << '\n'
         << hit_time_step << '\n'
         << gravity_device_id << ' ' << missile_cost << '\n';
}

void run_step(ParticleSystem& s, int step) {
    const int n = s.n;
    const double t = step * param::dt;

    // pre-calculate effective device masses
    std::vector<double> effective_masses = s.m;
    for (int id : s.device_ids)
        effective_masses[id] = get_device_mass(s.m[id], t);

    std::vector<double> ax(n, 0.0), ay(n, 0.0), az(n, 0.0);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) continue;

            double dx = s.qx[j] - s.qx[i];
            double dy = s.qy[j] - s.qy[i];
            double dz = s.qz[j] - s.qz[i];
            
            double r2 = dx*dx + dy*dy + dz*dz + param::eps*param::eps;
            double r = std::sqrt(r2);
            double r3 = r * r2;

            double f = param::G * effective_masses[j] / r3;
            ax[i] += f * dx;
            ay[i] += f * dy;
            az[i] += f * dz;
        }
    }

    for (int i = 0; i < n; i++) {
        s.vx[i] += ax[i] * param::dt;
        s.vy[i] += ay[i] * param::dt;
        s.vz[i] += az[i] * param::dt;

        s.qx[i] += s.vx[i] * param::dt;
        s.qy[i] += s.vy[i] * param::dt;
        s.qz[i] += s.vz[i] * param::dt;
    }
}

double solve_problem1(ParticleSystem s) {
    for (int id : s.device_ids)
        s.m[id] = 0;

    double min_dist = s.disSquared(s.planet, s.asteroid);
    
    for (int step = 1; step <= param::n_steps; step++) {
        run_step(s, step);
    
        double d2 = s.disSquared(s.planet, s.asteroid);
        if (d2 < min_dist)
            min_dist = d2;
    }
    return std::sqrt(min_dist);
}

double sovle_problem2(ParticleSystem s) {
    double r2 = param::planet_radius * param::planet_radius;
    if (s.disSquared(s.planet, s.asteroid) < r2)
        return 0;

    for (int step = 1; step <= param::n_steps; step++) {
        run_step(s, step);
    
        if (s.disSquared(s.planet, s.asteroid) < r2)
            return step;
    }
    return -2;
}

std::pair<int, double> solve_problem3(ParticleSystem& initial_s, int hit_time_step) {
    if (hit_time_step == -2)
        return {-1, 0.0};
    
    int best_device_id = -1;
    double min_missile_cost = std::numeric_limits<double>::infinity();
    double r2 = param::planet_radius * param::planet_radius;

    ParticleSystem s = initial_s;
    for (int id: initial_s.device_ids) {
        s.reset_to(initial_s);

        bool destroyed = false;
        int destroy_step = -1;
        bool hit_planet = false;

        for (int step = 0; step <= param::n_steps; step++) {
            // missle logic
            if (!destroyed) {
                double dist = s.distance(s.planet, id);
                double missle_travel_dist = param::missile_speed * (step+1) * param::dt;
                if (missle_travel_dist >= dist) {  // device destroyed
                    destroyed = true;
                    destroy_step = step;
                    s.m[id] = 0;
                }
            }

            if (step > 0) 
                run_step(s, step);

            // collision check
            if (s.disSquared(s.planet, s.asteroid) < r2) {
                hit_planet = true;
                break;  // fail attempt
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
        min_missile_cost = 0.0;

    return {best_device_id, min_missile_cost};
}

int main(int argc, char** argv) {
    if (argc != 3)
        throw std::runtime_error("must supply 2 arguments");

    ParticleSystem initial_s = read_input(argv[1]);

    double min_dist = solve_problem1(initial_s);
    int hit_time_step = sovle_problem2(initial_s);
    auto [best_device_id, min_missile_cost] = solve_problem3(initial_s, hit_time_step);

    write_output(argv[2], min_dist, hit_time_step, best_device_id, min_missile_cost);
    return 0;
}
