#include "PSA.h"
using namespace std;

PSA::PSA(Options opt, Graph gr) {
	//Constructor
	options = opt;
	graph = gr;
}

void PSA::run() {
	// Initializing multiple processes
	MPI_Init(NULL, NULL);
	// Get the number of processors
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	// Get the rank of the current processor
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Initialize a variable to measure the time taken
	clock_t begin;
	if (world_rank == 0) begin = clock(); // First processor is responsible for measuring

	// Intialize an arbitrary best tour
	int* bestTour = (int*)malloc(graph.n * sizeof(int));
	bestTour[0] = options.startNode;
	int offset = 0;
	for (int i = 0; i < graph.n; i++) {
		if (i == options.startNode) {
			offset = -1;
		}
		else {
			bestTour[i + 1 + offset] = i;
		}
	}

	// Shuffle the best tour
	default_random_engine g(options.randomState);
	for (auto i = graph.n - 1; i > 1; --i) {
		uniform_int_distribution<decltype(i)> d(1, i);
		int rand = d(g);
		swap(bestTour[i], bestTour[rand]);
	}

	// Calculate the cost of the current best tour
	double bestCost = calcCost(bestTour);

	//if (options.verbose >= 1 && world_rank == 0) {
	//	/*cout << "Initial tour: " << endl;;
	//	for (int i = 0; i < graph.n; i++) cout << bestTour[i] << " ";
	//	cout << endl;*/
	//	cout << "Initial cost: " << bestCost << endl << endl;
	//}

	int* tour = NULL;
	double cost;
	int* currentTour = (int*)malloc(graph.n * sizeof(int));
	double currentCost;
	double T;
	int iter2 = 0, iter1 = 0, totalIter = 0;
	double temperatureStep = (options.tempInit - options.tempThres) / options.exchangeIter;
	double nextReductionTemp = options.tempInit - temperatureStep;

	T = options.tempInit;
	// Loop while temperature is higher than a threshold
	while (T > options.tempThres) {
		// Copy the memory of bestTour to currentTour to start the searches with the common best tour
		memcpy(currentTour, bestTour, graph.n * sizeof(int));
		currentCost = bestCost;
		iter2++;

		if (tour != NULL) free(tour);
		// Obtain random next move and its cost
		tour = nextMove(currentTour);
		cost = calcCost(tour);
		// Check whether the move could be accepted
		if (accept(T, cost - currentCost)) {
			// Update the current tour if accepted
			memcpy(currentTour, tour, graph.n * sizeof(int));
			currentCost = cost;
			if (currentCost < bestCost) {
				// Update best tour if the current tour is better than previous tour
				memcpy(bestTour, currentTour, graph.n * sizeof(int));
				bestCost = currentCost;
			}
		}

		if (options.verbose >= 3) {
			printf("Processor %d achieves %f at temperature %f\n\n", world_rank,
				bestCost, T);
		}
		// Update temperature by subtracting a decay
		T -= options.tempAlpha;
		// Every processors broadcast their bestCost results to each other
		// The result is saved in the array, ranks
		if (T <= nextReductionTemp) {
			double* ranks = (double*)malloc(world_size * sizeof(double));
			for (int i = 0; i < world_size; i++) {
				if (world_rank == i) {
					ranks[i] = bestCost;
				}
				MPI_Bcast(&ranks[i], 1, MPI_DOUBLE, i, MPI_COMM_WORLD);
			}
			// Find the processor with the best result
			int bestProc = 0;
			for (int i = 0; i < world_size; i++) {
				if (ranks[i] < ranks[bestProc]) bestProc = i;
			}
			// Broadcast the best tour of the best processor to all other processors
			// In the next iteration, all processors will start the searches with the common best tour
			bestCost = ranks[bestProc];
			MPI_Bcast(bestTour, graph.n, MPI_INT, bestProc, MPI_COMM_WORLD);

			// Update the next temperature reduction point
			nextReductionTemp -= temperatureStep;
		}
	}
	if (options.verbose >= 1 && world_rank == 0) {
		/*cout << "Best tour: " << endl;;
		for (int i = 0; i < graph.n; i++) cout << bestTour[i] << " ";
		cout << endl;*/
		cout << "Best cost: " << bestCost << endl;
		totalIter = ((iter2 * options.equiIter) * options.exchangeIter);
		//cout << "Total iterations: " << totalIter << endl;
	}
	// First processor is responsible for measuring the time taken
	if (world_rank == 0) {
		clock_t end = clock();
		double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;
		cout << "Time spent: " << time_spent << "s" << endl;
	}
	// End the multiprocessing
	MPI_Finalize();
}

double PSA::norm(double std, double x) {
	// A probability function used to intialize different parameters for different processors
	return exp(-x * x / (2 * std * std));
}

void PSA::swap(int& a, int& b) {
	// Swap two integers
	int temp = a;
	a = b;
	b = temp;
}

bool PSA::accept(double T, double changeInCost) {
	if (changeInCost < 0) {
		// If next move is better, accept
		return true;
	}
	else if (exp(-changeInCost / T) > ((float)rand() / RAND_MAX)) {
		// If next move is worse, accept at a chance based on the temperature
		return true;
	}
	else {
		// Not accepted
		return false;
	}
}

double PSA::calcCost(int* tour) {
	// Calculate cost by adding all the weights between adjacent nodes
	double totalCost = 0;
	double cost;
	int u, v, j;
	for (int i = 0; i < graph.n; i++) {
		u = tour[i];
		if (i < graph.n - 1) j = i + 1;
		else j = 0;
		v = tour[j];
		// Get the cost of adjacent nodes and add to totalCost
		cost = graph.getWeight(u, v);
		totalCost += cost;
	}
	return totalCost;

}

int* PSA::nextMove(int* currentTour) {
	// Generate random next move by 2-opt method
	// Randomly select two indexes, first and second
	int* tour = (int*)malloc(graph.n * sizeof(int));
	memcpy(tour, currentTour, graph.n * sizeof(int));
	size_t size = graph.n;
	size_t first = rand() * (size - 1) / RAND_MAX;
	size_t second = rand() * (size - 1) / RAND_MAX;
	if (options.swapType == 2) //2 opt method swap
	{
		if (first == second) {
			first--;
		}
		if (second < first) {
			first = first + second;
			second = first - second;
			first = first - second;
		}
		second += 2;
		// Reverse the order of all the elements between first and second
		while ((first != second) && (first != --second)) {
			swap(tour[first], tour[second]);
			++first;
		}
		return tour;
	}
	else
		if (options.swapType == 1) //regular pair swap
		{
			swap(tour[first], tour[second]);
			return tour;
		}
		else
		{
			swap(tour[first], tour[second]);
			return tour;
		}
}

