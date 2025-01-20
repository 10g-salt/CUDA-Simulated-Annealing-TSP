#ifndef PARALLELSIMULATEDANNEALING_H
#define PARALLELSIMULATEDANNEALING_H

#include "Graph.h"
#include <iostream>
#include <cstring>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <mpi.h>

struct Options {
    double tempInit;
    double tempAlpha;
    double tempThres;
    int equiIter;
    int exchangeIter;
    int startNode;
    int verbose;
    int randomState;
    int swapType;
};

class PSA {
private:
    Options options;
    Graph graph;

public:
    PSA(Options opt, Graph gr);
    void run();
    double norm(double std, double x);
    void swap(int& a, int& b);
    bool accept(double T, double changeInCost);
    double calcCost(int* tour);
    int* nextMove(int* currentTour);
};

#endif /* PARALLELSIMULATEDANNEALING_H */
