batch
@echo off
echo Компиляция CUDA симулятора частиц...
nvcc -o particle_system.exe ../src/particle_system.cu -arch=sm_61 -O3
echo Готово!
