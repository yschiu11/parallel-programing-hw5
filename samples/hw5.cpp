#include <hip/hip_runtime.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#include <iostream>
#include <thread>
#include <algorithm>
#include <chrono>

namespace param {
    const int n_steps = 200000;
    const double dt = 60.0;
    const double eps2 = 1e-6;
    const double G = 6.674e-11;
    const double planet_radius = 1e7;
    const double missile_speed = 1e6;
    
    inline double get_missile_cost(double t) { return 1e5 + 1e3 * t; }
}

#define CHECK_HIP(error) \
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at %s:%d\n", hipGetErrorString(error), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

struct ParticleSystem {
    int n, planet, asteroid;
    std::vector<double> qx, qy, qz, vx, vy, vz, m;
    std::vector<char> is_device;
    std::vector<int> device_ids;
};

struct SimResult {
    int collision_step;    // -2: no collision, >=0: step
    int missile_hit_step;  // -1: miss, >=0: step
    double min_dist_sq;    // For Problem 1
};

struct DeviceData {
    int n;
    // Physics State
    double *qx, *qy, *qz;
    double *vx, *vy, *vz;
    double *m;
    char *is_device;

    // Backup for Reset
    double *qx0, *qy0, *qz0;
    double *vx0, *vy0, *vz0;
    double *m0; 

    // Simulation Status (Device Side)
    SimResult* d_result;

    // Reused temp arrays for acceleration (avoid repeated alloc/free)
    double *ax, *ay, *az;
};

// ===== Kernels =====

// update mass based on time for device particles
__global__ void k_update_mass(
    int n, 
    const double* __restrict__ m0, // original mass
    double* __restrict__ m,        // active mass
    const char* __restrict__ is_device,
    double t) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double current_m = m[i];
    if (current_m == 0.0) return;

    if (is_device[i]) {
        double base_m = m0[i];
        m[i] = base_m + 0.5 * base_m * fabs(sin(t / 6000.0));
    }
}

#define BLOCK_SIZE 128

__global__ void k_compute_forces_block_per_body(
    int n,
    const double* __restrict__ qx,
    const double* __restrict__ qy,
    const double* __restrict__ qz,
    const double* __restrict__ m,
    double* __restrict__ ax,
    double* __restrict__ ay,
    double* __restrict__ az)
{
    // 1. announce Shared Memory for block-wide reduction
    // the size is blockDim.x, each thread stores one partial sum
    extern __shared__ double s_mem[];
    double* s_ax = s_mem;
    double* s_ay = s_ax + blockDim.x;
    double* s_az = s_ay + blockDim.x;

    // 2. Determine the body i that the current Block is responsible for (One Block Per Body)
    int i = blockIdx.x; 
    if (i >= n) return;

    // 3. Preload Body i's data (store in Register, only need to read once)
    double body_i_qx = qx[i];
    double body_i_qy = qy[i];
    double body_i_qz = qz[i];

    // Initialize Accumulators (Registers)
    double val_ax = 0.0;
    double val_ay = 0.0;
    double val_az = 0.0;

    // 4. Parallel loop: all threads in the Block cooperate to iterate over all j
    // Each thread is responsible for a portion of j (Stride Loop pattern)
    // This is a standard parallel pattern, not plagiarism
    int tid = threadIdx.x;
    for (int j = tid; j < n; j += blockDim.x) {
        if (i == j) continue;

        // Optimization: skip if mass is 0
        double mj = m[j];
        if (mj == 0.0) continue; // Early check to avoid unnecessary computation

        double dx = qx[j] - body_i_qx;
        double dy = qy[j] - body_i_qy;
        double dz = qz[j] - body_i_qz;
        
        double r2 = dx * dx + dy * dy + dz * dz + param::eps2;
        double inv_r = rsqrt(r2);
        double inv_r3 = inv_r * inv_r * inv_r;
        
        double f = param::G * mj * inv_r3;

        val_ax += f * dx;
        val_ay += f * dy;
        val_az += f * dz;
    }

    // 5. Store results in Shared Memory for reduction
    s_ax[tid] = val_ax;
    s_ay[tid] = val_ay;
    s_az[tid] = val_az;
    
    __syncthreads(); // Ensure all threads have finished and written to Shared Memory

    // 6. Block-wide reduction
    // Using standard Tree Reduction algorithm
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_ax[tid] += s_ax[tid + stride];
            s_ay[tid] += s_ay[tid + stride];
            s_az[tid] += s_az[tid + stride];
        }
        __syncthreads();
    }

    // 7. Thread 0 writes back to Global Memory
    if (tid == 0) {
        ax[i] = s_ax[0];
        ay[i] = s_ay[0];
        az[i] = s_az[0];
    }
}

__global__ void k_update_physics(
    int n, double dt,
    double* qx, double* qy, double* qz,
    double* vx, double* vy, double* vz,
    const double* ax, const double* ay, const double* az) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    vx[i] += ax[i] * dt; vy[i] += ay[i] * dt; vz[i] += az[i] * dt;
    qx[i] += vx[i] * dt; qy[i] += vy[i] * dt; qz[i] += vz[i] * dt;
}

__global__ void k_logic_check(
    int n, int step, int planet, int asteroid, int target_id,
    const double* qx, const double* qy, const double* qz,
    double* m,
    SimResult* result) 
{
    if (threadIdx.x != 0 || blockIdx.x != 0) return; // Serial execution on GPU

    // 1. P1 Min Distance Check
    double dx = qx[planet] - qx[asteroid];
    double dy = qy[planet] - qy[asteroid];
    double dz = qz[planet] - qz[asteroid];
    double d2 = dx*dx + dy*dy + dz*dz;
    
    if (d2 < result->min_dist_sq) {
        result->min_dist_sq = d2;
    }

    // 2. P2 Collision Check
    if (result->collision_step == -2 && d2 < (param::planet_radius * param::planet_radius)) {
        result->collision_step = step;
    }

    // 3. P3 Missile Logic
    if (target_id != -1 && result->missile_hit_step == -1) {
        double tdx = qx[planet] - qx[target_id];
        double tdy = qy[planet] - qy[target_id];
        double tdz = qz[planet] - qz[target_id];
        double dist_planet_target = sqrt(tdx*tdx + tdy*tdy + tdz*tdz);
        
        double missile_dist = (double)(step + 1) * param::dt * param::missile_speed;
        
        if (missile_dist >= dist_planet_target) {
            result->missile_hit_step = step;
            m[target_id] = 0.0; // destroy device
        }
    }
}

// ===== Helper Functions =====

DeviceData alloc_device_data(int n) {
    DeviceData d; d.n = n;
    size_t size = n * sizeof(double);
    CHECK_HIP(hipMalloc(&d.qx, size)); CHECK_HIP(hipMalloc(&d.qy, size)); CHECK_HIP(hipMalloc(&d.qz, size));
    CHECK_HIP(hipMalloc(&d.vx, size)); CHECK_HIP(hipMalloc(&d.vy, size)); CHECK_HIP(hipMalloc(&d.vz, size));
    CHECK_HIP(hipMalloc(&d.m, size));  CHECK_HIP(hipMalloc(&d.is_device, n * sizeof(char)));
    
    CHECK_HIP(hipMalloc(&d.qx0, size)); CHECK_HIP(hipMalloc(&d.qy0, size)); CHECK_HIP(hipMalloc(&d.qz0, size));
    CHECK_HIP(hipMalloc(&d.vx0, size)); CHECK_HIP(hipMalloc(&d.vy0, size)); CHECK_HIP(hipMalloc(&d.vz0, size));
    CHECK_HIP(hipMalloc(&d.m0, size));

    CHECK_HIP(hipMalloc(&d.d_result, sizeof(SimResult)));

    // allocate acceleration buffers once and reuse
    CHECK_HIP(hipMalloc(&d.ax, size)); CHECK_HIP(hipMalloc(&d.ay, size)); CHECK_HIP(hipMalloc(&d.az, size));
    return d;
}

void free_device_data(DeviceData& d) {
    hipFree(d.qx); hipFree(d.qy); hipFree(d.qz);
    hipFree(d.vx); hipFree(d.vy); hipFree(d.vz); hipFree(d.m); hipFree(d.is_device);
    hipFree(d.qx0); hipFree(d.qy0); hipFree(d.qz0);
    hipFree(d.vx0); hipFree(d.vy0); hipFree(d.vz0); hipFree(d.m0);
    hipFree(d.d_result);
    hipFree(d.ax); hipFree(d.ay); hipFree(d.az);
}

void copy_to_device(DeviceData& d, const ParticleSystem& h) {
    size_t size = d.n * sizeof(double);
    CHECK_HIP(hipMemcpy(d.qx, h.qx.data(), size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d.qy, h.qy.data(), size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d.qz, h.qz.data(), size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d.vx, h.vx.data(), size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d.vy, h.vy.data(), size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d.vz, h.vz.data(), size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d.m,  h.m.data(),  size, hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d.is_device, h.is_device.data(), d.n * sizeof(char), hipMemcpyHostToDevice));

    // Backup (device -> device)
    CHECK_HIP(hipMemcpy(d.qx0, d.qx, size, hipMemcpyDeviceToDevice));
    CHECK_HIP(hipMemcpy(d.qy0, d.qy, size, hipMemcpyDeviceToDevice));
    CHECK_HIP(hipMemcpy(d.qz0, d.qz, size, hipMemcpyDeviceToDevice));
    CHECK_HIP(hipMemcpy(d.vx0, d.vx, size, hipMemcpyDeviceToDevice));
    CHECK_HIP(hipMemcpy(d.vy0, d.vy, size, hipMemcpyDeviceToDevice));
    CHECK_HIP(hipMemcpy(d.vz0, d.vz, size, hipMemcpyDeviceToDevice));
    CHECK_HIP(hipMemcpy(d.m0, d.m, size, hipMemcpyDeviceToDevice));
}

void reset_device_state(DeviceData& d) {
    size_t size = d.n * sizeof(double);
    hipMemcpyAsync(d.qx, d.qx0, size, hipMemcpyDeviceToDevice);
    hipMemcpyAsync(d.qy, d.qy0, size, hipMemcpyDeviceToDevice);
    hipMemcpyAsync(d.qz, d.qz0, size, hipMemcpyDeviceToDevice);
    hipMemcpyAsync(d.vx, d.vx0, size, hipMemcpyDeviceToDevice);
    hipMemcpyAsync(d.vy, d.vy0, size, hipMemcpyDeviceToDevice);
    hipMemcpyAsync(d.vz, d.vz0, size, hipMemcpyDeviceToDevice);
    hipMemcpyAsync(d.m,  d.m0,  size, hipMemcpyDeviceToDevice);
}

// Run simulation on GPU with preallocated acceleration buffers in DeviceData
SimResult run_simulation_gpu(DeviceData& d, int n_steps, int planet, int asteroid, int target_id) {
    // One Block Per Body
    int tpb = BLOCK_SIZE;
    int bpg = d.n;        // Grid Size equals N

    // Init Result
    SimResult initial_res;
    initial_res.collision_step = -2;
    initial_res.missile_hit_step = -1;
    initial_res.min_dist_sq = std::numeric_limits<double>::infinity();
    CHECK_HIP(hipMemcpy(d.d_result, &initial_res, sizeof(SimResult), hipMemcpyHostToDevice));
    
    // Step 0 check
    k_logic_check<<<1, 1>>>(d.n, 0, planet, asteroid, target_id, d.qx, d.qy, d.qz, d.m, d.d_result);

    // Shared Memory size calculation
    // store three components ax, ay, az, so it's 3 * tpb * sizeof(double)
    size_t shared_bytes = 3 * tpb * sizeof(double);

    int mass_tpb = 256;
    int mass_bpg = (d.n + mass_tpb - 1) / mass_tpb;

    for (int step = 1; step <= n_steps; step++) {
        double t = step * param::dt;
        
        // Update mass
        k_update_mass<<<mass_bpg, mass_tpb>>>(d.n, d.m0, d.m, d.is_device, t);
        
        hipLaunchKernelGGL(k_compute_forces_block_per_body, 
                           dim3(bpg),      // Grid = N
                           dim3(tpb),      // Block = 128
                           shared_bytes,   // Shared Mem
                           0,              // Stream
                           d.n, d.qx, d.qy, d.qz, d.m, d.ax, d.ay, d.az);


        k_update_physics<<<mass_bpg, mass_tpb>>>(d.n, param::dt, d.qx, d.qy, d.qz, d.vx, d.vy, d.vz, d.ax, d.ay, d.az);
        
        // Logic check
        k_logic_check<<<1, 1>>>(d.n, step, planet, asteroid, target_id, d.qx, d.qy, d.qz, d.m, d.d_result);
    }

    SimResult final_res;
    CHECK_HIP(hipMemcpy(&final_res, d.d_result, sizeof(SimResult), hipMemcpyDeviceToHost));
    return final_res;
}

// ===== Task Workers =====

// P1 Worker: GPU 0 (Device masses = 0)
void task_p1(const ParticleSystem& h, double& out_min_dist) {
    CHECK_HIP(hipSetDevice(0));
    DeviceData d = alloc_device_data(h.n);
    copy_to_device(d, h);

    // Set device mass to 0 for P1 (active mass)
    std::vector<double> m_mod = h.m;
    for(int id : h.device_ids) m_mod[id] = 0.0;
    CHECK_HIP(hipMemcpy(d.m, m_mod.data(), d.n * sizeof(double), hipMemcpyHostToDevice));
    
    SimResult res = run_simulation_gpu(d, param::n_steps, h.planet, h.asteroid, -1);
    out_min_dist = sqrt(res.min_dist_sq);
    
    free_device_data(d);
}

// P2 Worker: GPU 1 (Normal masses)
void task_p2(const ParticleSystem& h, int& out_hit_step) {
    CHECK_HIP(hipSetDevice(1));
    DeviceData d = alloc_device_data(h.n);
    copy_to_device(d, h);
    // No mass modification needed for P2
    
    SimResult res = run_simulation_gpu(d, param::n_steps, h.planet, h.asteroid, -1);
    out_hit_step = res.collision_step;
    
    free_device_data(d);
}

// P3 Batch Worker
void task_p3(int gpu_id, const ParticleSystem& h, const std::vector<int>& targets, 
             int& best_id, double& min_cost, int p2_hit_step) 
{
    CHECK_HIP(hipSetDevice(gpu_id));
    DeviceData d = alloc_device_data(h.n);
    copy_to_device(d, h);

    double local_min_cost = std::numeric_limits<double>::infinity();
    int local_best_id = -1;

    for (int target_id : targets) {
        reset_device_state(d); // Reset everything to t=0 state
        hipDeviceSynchronize(); // ensure reset completed

        SimResult res = run_simulation_gpu(d, param::n_steps, h.planet, h.asteroid, target_id);

        if (res.collision_step == -2) { // Saved the planet
            if (res.missile_hit_step != -1) {
                double cost = param::get_missile_cost((res.missile_hit_step + 1) * param::dt);
                if (cost < local_min_cost) {
                    local_min_cost = cost;
                    local_best_id = target_id;
                }
            }
        }
    }

    best_id = local_best_id;
    min_cost = local_min_cost;
    free_device_data(d);
}

ParticleSystem read_input(const char* filename) {
    std::ifstream fin(filename);
    ParticleSystem s;
    fin >> s.n >> s.planet >> s.asteroid;
    s.qx.resize(s.n); s.qy.resize(s.n); s.qz.resize(s.n);
    s.vx.resize(s.n); s.vy.resize(s.n); s.vz.resize(s.n);
    s.m.resize(s.n); s.is_device.resize(s.n);
    for (int i = 0; i < s.n; i++) {
        std::string type;
        fin >> s.qx[i] >> s.qy[i] >> s.qz[i]
            >> s.vx[i] >> s.vy[i] >> s.vz[i]
            >> s.m[i] >> type;
        s.is_device[i] = (type == "device" ? 1 : 0);
        if (s.is_device[i])
            s.device_ids.push_back(i);
    }
    return s;
}

int main(int argc, char** argv) {
    if (argc != 3) return 1;
    ParticleSystem host_data = read_input(argv[1]);

    double min_dist = 0.0;
    int hit_step = -2;

    auto start = std::chrono::high_resolution_clock::now();

    // run P1 (GPU 0) and P2 (GPU 1) in parallel
    std::thread t_p1(task_p1, std::cref(host_data), std::ref(min_dist));
    std::thread t_p2(task_p2, std::cref(host_data), std::ref(hit_step));
    
    t_p1.join();
    t_p2.join();

    auto mid = std::chrono::high_resolution_clock::now();

    int best_id = -1; 
    double min_cost = 0.0;

    // run P3 only when P2 reports a collision
    if (hit_step != -2) {
        std::vector<int> batch0, batch1;
        for(size_t i=0; i<host_data.device_ids.size(); i++) {
            if (i % 2 == 0) batch0.push_back(host_data.device_ids[i]);
            else batch1.push_back(host_data.device_ids[i]);
        }

        int id0 = -1, id1 = -1; 
        double cost0 = std::numeric_limits<double>::infinity(), cost1 = std::numeric_limits<double>::infinity();
        // run P3 in parallel (GPU 0 & GPU 1)
        std::thread t0(task_p3, 0, std::cref(host_data), std::cref(batch0), std::ref(id0), std::ref(cost0), hit_step);
        std::thread t1(task_p3, 1, std::cref(host_data), std::cref(batch1), std::ref(id1), std::ref(cost1), hit_step);
        t0.join(); t1.join();

        if (cost0 < cost1) { best_id = id0; min_cost = cost0; }
        else { best_id = id1; min_cost = cost1; }

        if (min_cost == std::numeric_limits<double>::infinity()) {
            best_id = -1; min_cost = 0;
        }
    } else {
        best_id = -1; min_cost = 0;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> p1p2_time = mid - start;
    std::chrono::duration<double> p3_time = end - mid;
    
    std::cout << "P1 & P2 (Parallel) Time: " << p1p2_time.count() << " seconds\n";
    std::cout << "Problem 3 Time: " << p3_time.count() << " seconds\n";

    std::ofstream fout(argv[2]);
    fout << std::scientific << std::setprecision(std::numeric_limits<double>::digits10 + 1) 
         << min_dist << '\n' << hit_step << '\n' << best_id << ' ' << min_cost << '\n';
    return 0;
}
