#ifndef GRAPH_H
#define GRAPH_H

class Graph {
private:
    double** weights; // 2D array to store weights between nodes

public:
    int n; // Number of nodes
    Graph();
    Graph(int numNodes); // Constructor
    void setWeight(int u, int v, double weight); // Set weight between nodes u and v
    double getWeight(int u, int v); // Get weight between nodes u and v
};

#endif /* GRAPH_H */
