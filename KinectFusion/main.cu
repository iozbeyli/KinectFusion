#include "Visualizer.cuh"

int main()
{  
    Visualizer::getInstance()->run();   
    Visualizer::deleteInstance();
}