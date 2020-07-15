#include "Visualizer.h"

int main()
{  
    Visualizer::getInstance()->run();   
    Visualizer::deleteInstance();
}