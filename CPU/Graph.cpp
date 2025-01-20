#include "Graph.h"
#include <iostream>

// Constructor
Graph::Graph() {
}

Graph::Graph(int numNodes) {
    n = numNodes;
    // Allocate memory for the 2D array
    weights = new double* [n];
    for (int i = 0; i < n; ++i) {
        weights[i] = new double[n];
        // Initialize weights to 0 (assuming no edge initially)
        for (int j = 0; j < n; ++j) {
            weights[i][j] = 0.0;
        }
    }
}

// Set weight between nodes u and v
void Graph::setWeight(int u, int v, double weight) {
    if (u >= 0 && u < n && v >= 0 && v < n) {
        weights[u][v] = weight;
        weights[v][u] = weight; // Assuming an undirected graph
    }
    else {
        std::cerr << "Invalid node indices!" << std::endl;
    }
}

// Get weight between nodes u and v
double Graph::getWeight(int u, int v) {
    if (u >= 0 && u < n && v >= 0 && v < n) {
        return weights[u][v];
    }
    else {
        std::cerr << "Invalid node indices!" << std::endl;
        return 0.0; // Return default weight if indices are invalid
    }
}