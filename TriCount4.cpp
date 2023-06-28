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
    unordered_map<int, unordered_set<int>> adj;

    // Function to add an edge to graph
    void addEdge(int v, int w);

    // Add edge based on NodeIterator++
    // The smaller Node has the higher node in its neighbor list
    void addEdgeNIP(int v, int w);
 
    // Returns number of triangles in the graph
    // Pass the time variable by reference to capture the computing time
    int triCount(double& time);
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

    int maxNum = max(v,w);
    int minNum = min(v,w);

    // if the data doesn't already exist in the hashtable
    // put it into the hashtable

    adj[minNum].insert(maxNum);

    if(maxNum > vertices){
        vertices = maxNum;
    }

}

int Graph::triCount(double &time)
{

    int result;

    //cout << "Triangles In The Graph:" << endl;

    // Calculate the time for computing the triangle count
    Kokkos::Timer timer;

    Kokkos::parallel_reduce("outer loop", vertices, KOKKOS_LAMBDA (int a, int& reduction){      
    //for(int a = 0; a < vertices -1;a++){      

        auto index = adj.find(a); 
  
        if(index != adj.end()){    
            const unordered_set<int>& set = index->second;

            for(const auto &pair: set){

                auto index2 = adj.find(pair);

                if(index2 != adj.end()){
                    const unordered_set<int>& set2 = index2->second;

                    for(const auto &pair2 : set2){
                        if(set.find(pair2) != set.end()){

                            // output the triangle
                            // cout << "Triangles: " << a << " -> " << b << " -> " << pair << endl;

                            reduction++;
                            
                        }
                    }
                }
                    
            }
        }               


    }, result);

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

            // g.addEdgeNIP(3, 4);
            g.addEdgeNIP(firstInt,secondInt);

        }else{
            cout << "Failed to extract integers from line" << endl;
        }
    }
 

    // Function call
    double time;
    int triCountResult = g.triCount(time);

    cout << "Core Count: " << core_count << endl;
    cout << "Graph Dataset: " << inputString << endl;
    cout << "Triangle Count: " << triCountResult << endl;
    cout << "Time To Compute: " << time << " seconds" << endl;

    Kokkos::finalize();
 
    return 0;



}
