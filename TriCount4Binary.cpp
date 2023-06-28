#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include <list>
#include <unordered_map>
#include <algorithm>

#include <sstream>
#include <fstream>

#include <Kokkos_Core.hpp>

using namespace std;
 
// Graph class to represent an undirected graph
// using adjacency list representation
class Graph {

public:

    // Constructor of the class
    // if the size of the network is known this constructor can be used
    // Graph(int size);

    // Vertex count
    int vertices = -1; 
    
    // Neighbor list
    vector<vector<int>> adj;

    // Function to add an edge to graph
    void addEdge(int v, int w);

    // Add edge based on NodeIterator++
    // The smaller Node has the higher node in its neighbor list
    void addEdgeNIP(int v, int w);
 
    // Returns number of triangles in the graph
    // Pass the time variable by reference to capture the computing time
    int triCount(double& time);

    void sortNetwork(double& time);
};

// Graph::Graph(int size){

//     vertices = size;

// }
 
void Graph::addEdge(int v, int w)
{
    // Add w to v’s list.
    // adj[v].push_back(w);

    // Add w to v’s list.
    // adj[w].push_back(v);
}

void Graph::addEdgeNIP(int v, int w){

    // v is small - w is large
    adj[v].push_back(w);

}

void Graph::sortNetwork(double &time){

    Kokkos::Timer sortTimer;

    for(int a = 0; a < vertices; a++){
        int size = adj[a].size();
        for(int b = 0; b < size; b++){
            sort(adj[a].begin(), adj[a].end());
        }
    }

    time = sortTimer.seconds();

}

int Graph::triCount(double &time)
{

    // Let kokkos decide to whether use OpenMP, Cuda or other existing methods
    // You can change the execution space according to your program's needs
    // using ExecutionSpace = Kokkos::DefaultExecutionSpace;
    
    // using RangePolicy = Kokkos::RangePolicy<ExecutionSpace>;

    int result = 0;  

    // Calculate the time for computing the triangle count
    Kokkos::Timer timer;

    Kokkos::parallel_reduce( "outer loop", vertices, KOKKOS_LAMBDA (int a, int& reduction) {
    //for(int a = 0; a < vertices -1;a++){

        // use ThreadVectorRange or TeamThreadRange if you want to use nested reductions
        //Kokkos::parallel_reduce("inner reduction", size, KOKKOS_LAMBDA (int b, int& inner_reduction){
            for(int b =0 ; b< adj[a].size(); b++){

            int element = adj[a][b];

            // 1 = {0,2}
            
            // if perfectly sorted you can use this
            // int i = b; otherwise i = 0;
            int i = b;

            int k = 0;

            while(i < adj[a].size() && k < adj[element].size()){

                if(adj[a][i] > adj[element][k]){
                    k++;
                }else if(adj[element][k] > adj[a][i]){
                    i++;
                }else{
                    i++;
                    k++;
                    reduction++;
                }
            }

        }
        

    }, result);
    //}


    // Return the calculated time by reference
    time = timer.seconds();

    return result;

}
 
// Driver code
int main(int argc, char* argv[])
{

    // Command line processing
    if(argc != 2){
        cout << "Usage ./TriCount4.host <Input File>" << endl;
        exit(1);
    }

    string inputString = argv[1];



    Kokkos::initialize();

    // Get the core count to print at the end
    int core_count = Kokkos::HostSpace::execution_space::concurrency();

    // Initialize the graph and add edges from the input file
    Graph g;

    //g.adj.resize(1134890);
    g.adj.resize(1000);
    g.vertices = 1000;

    int maxNum, minNum;

    string line;

    ifstream inputFile(inputString);

    // Skip the first 4 lines if the dataset is friendster
    if(inputString == "com-friendster.ungraph.txt"){
        for(int a = 0; a < 4; a++){
            getline(inputFile, line);
        }
    }

    if(!inputFile){
        cout << "Failed to open the file" << endl;
        exit(1);
    }

    while (getline(inputFile, line))
    {
        istringstream iss(line);
        int firstInt, secondInt;

        if(iss >> firstInt >> secondInt){

            int maxNum = max(firstInt,secondInt);
            int minNum = min(firstInt,secondInt);

            if(maxNum > g.vertices){
                g.adj.resize(maxNum);
                g.vertices = maxNum;
            }
            // g.addEdgeNIP(3, 4);
            g.addEdgeNIP(minNum,maxNum);

        }else{
            cout << "Failed to extract integers from line" << endl;
        }
    }

    // Sort the graph
    double timeSort;
    g.sortNetwork(timeSort);

    // Function call
    double time;
    int triCountResult = g.triCount(time);

    cout << "Core Count: " << core_count << endl;
    cout << "Graph Dataset: " << inputString << endl;
    cout << "Triangle Count: " << triCountResult << endl;
    cout << "Time To Compute: " << time << " seconds" << endl;
    cout << "Time To Sort: " << timeSort << " seconds" << endl;

    Kokkos::finalize();
 
    return 0;



}
