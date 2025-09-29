#include <iostream>
#include <cuda_runtime.h>
#include <math.h>
#include <chrono>
#include <cstdlib>
#include <ctime>

struct Particle {
    float x, y;
    float vx, vy;
    float r, g, b;
};

// 1. Основная функция обновления частиц
__global__ void updateParticles(Particle* particles, int numParticles, float dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numParticles) return;

    // Применение гравитации
    particles[idx].vy -= 9.8f * dt;
    
    // Обновление позиции
    particles[idx].x += particles[idx].vx * dt;
    particles[idx].y += particles[idx].vy * dt;
    
    // Отскок от земли
    if (particles[idx].y < 0) {
        particles[idx].y = 0;
        particles[idx].vy = -particles[idx].vy * 0.8f;
    }
    
    // Отскок от стен
    if (particles[idx].x < -10 || particles[idx].x > 10) {
        particles[idx].vx = -particles[idx].vx * 0.8f;
        particles[idx].x = (particles[idx].x < -10) ? -10 : 10;
    }
}

// 2. Улучшенная функция обработки столкновений
__global__ void handleParticleCollisions(Particle* particles, int numParticles) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numParticles) return;

    float radius = 0.3f;
    float restitution = 0.9f; // Коэффициент восстановления
    
    // Проверяем только близлежащие частицы (для оптимизации производительности)
    for (int j = idx + 1; j < min(idx + 50, numParticles); j++) {
        float dx = particles[idx].x - particles[j].x;
        float dy = particles[idx].y - particles[j].y;
        float distance = sqrtf(dx*dx + dy*dy);
        
        if (distance < radius * 2 && distance > 0.001f) {
            // Единичный нормальный вектор
            float nx = dx / distance;
            float ny = dy / distance;
            
            // Относительная скорость
            float dvx = particles[idx].vx - particles[j].vx;
            float dvy = particles[idx].vy - particles[j].vy;
            float velocity_along_normal = dvx * nx + dvy * ny;
            
            // Не сталкиваемся, если частицы удаляются друг от друга
            if (velocity_along_normal > 0) continue;
            
            // Импульс столкновения
            float impulse = -(1.0f + restitution) * velocity_along_normal;
            impulse /= 2.0f;
            
            // Применение импульса
            particles[idx].vx += impulse * nx;
            particles[idx].vy += impulse * ny;
            particles[j].vx -= impulse * nx;
            particles[j].vy -= impulse * ny;
            
            // Коррекция перекрытия
            float overlap = radius * 2 - distance;
            particles[idx].x += nx * overlap * 0.5f;
            particles[idx].y += ny * overlap * 0.5f;
            particles[j].x -= nx * overlap * 0.5f;
            particles[j].y -= ny * overlap * 0.5f;
        }
    }
}

// 3. Функция сопротивления воздуха (опционально)
__global__ void applyAirResistance(Particle* particles, int numParticles, float dt) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= numParticles) return;

    float airResistance = 0.99f; // Простое сопротивление воздуха
    
    particles[idx].vx *= airResistance;
    particles[idx].vy *= airResistance;
}

// 4. Вспомогательные функции для управления памятью
void copyToDevice(Particle* d_particles, Particle* h_particles, int numParticles) {
    cudaMemcpy(d_particles, h_particles, numParticles * sizeof(Particle), cudaMemcpyHostToDevice);
}

void copyToHost(Particle* h_particles, Particle* d_particles, int numParticles) {
    cudaMemcpy(h_particles, d_particles, numParticles * sizeof(Particle), cudaMemcpyDeviceToHost);
}

int main() {
    const int NUM_PARTICLES = 3000;
    const float DT = 0.016f;
    
    // Initialize random seed
    std::srand(static_cast<unsigned>(std::time(0)));
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 1. Создание частиц на CPU
    Particle* h_particles = new Particle[NUM_PARTICLES];
    for (int i = 0; i < NUM_PARTICLES; i++) {
        h_particles[i] = {
            (float)(rand() % 15 - 7),   // x: -7 to 7
            (float)(rand() % 8 + 2),    // y: 2 to 10
            (float)(rand() % 6 - 3),    // vx: -3 to 3
            (float)(rand() % 4 - 6),    // vy: -6 to -2
            (float)rand() / RAND_MAX,
            (float)rand() / RAND_MAX,  
            (float)rand() / RAND_MAX
        };
    }
    
    // 2. Выделение памяти на GPU
    Particle* d_particles;
    cudaMalloc(&d_particles, NUM_PARTICLES * sizeof(Particle));
    
    // 3. Копирование данных на GPU
    copyToDevice(d_particles, h_particles, NUM_PARTICLES);
    
    std::cout << "Simulating " << NUM_PARTICLES << " particles with advanced physics...\n";
    
    // 4. Настройка блоков и потоков
    int blockSize = 256;
    int numBlocks = (NUM_PARTICLES + blockSize - 1) / blockSize;
    
    // 5. Основное моделирование
    for (int step = 0; step < 100; step++) {
        // Основное обновление движения
        updateParticles<<<numBlocks, blockSize>>>(d_particles, NUM_PARTICLES, DT);
        cudaDeviceSynchronize();
        
        // Сопротивление воздуха (каждые 3 шага)
        if (step % 3 == 0) {
            applyAirResistance<<<numBlocks, blockSize>>>(d_particles, NUM_PARTICLES, DT);
            cudaDeviceSynchronize();
        }
        
        // Столкновения между частицами (каждый шаг)
        handleParticleCollisions<<<numBlocks, blockSize>>>(d_particles, NUM_PARTICLES);
        cudaDeviceSynchronize();
        
        if (step % 20 == 0) {
            std::cout << "Step " << step << " - Physics simulation running...\n";
        }
    }
    
    // 6. Получение результатов с GPU
    copyToHost(h_particles, d_particles, NUM_PARTICLES);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // 7. Вывод результатов
    std::cout << "\n=== SIMULATION RESULTS ===" << std::endl;
    std::cout << "Execution time: " << duration.count() << " ms" << std::endl;
    std::cout << "First particle position: " << h_particles[0].x << ", " << h_particles[0].y << std::endl;
    
    // Дополнительный анализ
    float avgX = 0, avgY = 0;
    for (int i = 0; i < NUM_PARTICLES; i++) {
        avgX += h_particles[i].x;
        avgY += h_particles[i].y;
    }
    avgX /= NUM_PARTICLES;
    avgY /= NUM_PARTICLES;
    
    std::cout << "Average position: " << avgX << ", " << avgY << std::endl;
    std::cout << "Physics simulation completed successfully!" << std::endl;
    
    // 8. Очистка
    delete[] h_particles;
    cudaFree(d_particles);
    
    return 0;
}
