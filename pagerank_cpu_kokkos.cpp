#include <list>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <map>

#include <iostream>
#include <sstream>
#include <fstream>

//include <omp.h>

#include <Kokkos_Core.hpp>

using namespace std;

int main(int argc, char* argv[]) {
    
    // Command line processing
    if(argc != 2){
        cout << "Usage ./executable <Input File>" << endl;
        exit(1);
    }

    string inputString = argv[1];


    Kokkos::initialize();

    vector<vector<int>> adj;
    int vertices = 0;
    int edges = 0;

    string line;

    ifstream inputFile(inputString);

    if(!inputFile){
        //cout << "Failed to open the file" << endl;
        exit(1);
    }

    //printf("section 1");

    int maxNum;
    int firstInt, secondInt;

    while (getline(inputFile, line))
    {
        // check the commented lines    
        if(line[0] != '%'){    
            istringstream iss(line);

            if(iss >> firstInt >> secondInt){

                maxNum = max(firstInt,secondInt);

                if(maxNum >= vertices){
                    adj.resize(maxNum + 1);
                    vertices = maxNum + 1;
                }
                
                adj[firstInt].push_back(secondInt);
                //adj[secondInt].push_back(firstInt);

                edges++;
                

            }else{
                cout << "Failed to extract integers from line" << endl;
            }

        }
    }

    #ifdef KOKKOS_ENABLE_CUDA
    #define MemSpace Kokkos::CudaSpace
    #endif
    #ifdef KOKKOS_ENABLE_OPENMPTARGET
    #define MemSpace Kokkos::OpenMPTargetSpace
    #endif
    #ifndef MemSpace
    #define MemSpace Kokkos::HostSpace
    #endif

    using ExecSpace = MemSpace::execution_space;
    using range_policy = Kokkos::RangePolicy<ExecSpace>;
    //using range_policy = Kokkos::RangePolicy< Kokkos::Schedule<Kokkos::Dynamic>, ExecSpace >;

    // Allocate y, x vectors and Matrix A on device.
    //typedef Kokkos::View<int*, Kokkos::LayoutLeft, MemSpace>   ViewVectorType;
    typedef Kokkos::View<int*, MemSpace>   ViewVectorType;
    typedef Kokkos::View<float*, MemSpace>   ViewVectorTypeFloat;


    //printf("section 2\n");

    // populate the out_degree array
    // int out_degree[vertices];
    ViewVectorType out_degree("degree of each node array" , vertices);

    for(int a = 0; a < vertices; a++){
        out_degree(a) = adj[a].size();
    }

    // populate the in_neigh array
    vector<vector<int>> in_neigh;
    in_neigh.resize(vertices);

    for(int a = 0; a < vertices; a++){
        for(int b = 0; b < adj[a].size(); b++){
            int out_neighbor = adj[a][b];
            in_neigh[out_neighbor].push_back(a);
        }
    }

    // print to test
    for(int a = 0; a < vertices; a++){
        cout << "neigh of " << a << ": " ;
        for(int b = 0; b < adj[a].size(); b++){
            cout << adj[a][b] << " ";
        }
        cout << " " << endl;
    }
    cout << " " << endl;
    
    cout << "out degree list: " ;
    for(int a = 0; a < vertices; a++){
        cout << out_degree(a) << " ";
    }
    cout << " " << endl;

    cout << " " << endl;

    for(int a = 0; a < vertices; a++){
	cout << "in_neigh of " << a << ": " ;
        for(int b = 0; b < in_neigh[a].size(); b++){
            cout << in_neigh[a][b] << " ";
        }
	cout << " " << endl;
    }
    cout << " " << endl;
    
    
    // convert in_neigh 2d vector to CSR
    //vector<int> in_neigh_indexPointer;
    //vector<int> in_neigh_indices;

    ViewVectorType in_neigh_indexPointer("index pointer view of in_neigh list", vertices +1);
    ViewVectorType in_neigh_indices("indices pointer view of in_neigh list", edges);


    // index for indexPointer vector
    int i = 0;

    // index for indices vector
    //int j = 0;

    int count = 0;

    //indexPointer[0] = 0;
    in_neigh_indexPointer(0) = 0;

    // Populate the CSR
    for(int a = 0; a < vertices; a++){

        // Populate the indexPointer vector
        int size = in_neigh[a].size();
        i += size;

        //indexPointer[a+1] = i;
        in_neigh_indexPointer(a+1) = i;

        // Populate the indices vector
        for(int b = 0; b < size; b++){
            in_neigh_indices(count) = in_neigh[a][b];
            count++;
        }
    }

    cout << "lists of in_neigh by each element" << endl;
    for(int u = 0 ; u < vertices; u++){

    	int v = in_neigh_indexPointer(u);
	    int limit = in_neigh_indexPointer(u+1);
	
	cout << "element " << u  << " : " ;

	for(; v < limit; v++){
	   int v_element = in_neigh_indices(v);
	   cout << v_element << " " ;
	}
	cout << " " << endl;
    }

    cout << " " << endl;


    // Allocation done - Pagerank algorithm
 
    const float kDamp = 0.85;

    // everyone start with the same value 1.0/vertices
    const float init_score = 1.0f / vertices;
    const float base_score = (1.0f - kDamp) / vertices;

    //vector<float> scores(vertices, init_score);
    ViewVectorTypeFloat scores("scores view", vertices);

    // start each value with init_score
    for(int i = 0 ; i < vertices; i++){
        scores(i) = init_score;
    }

    //vector<float> outgoing_contrib(vertices);
    ViewVectorTypeFloat outgoing_contrib("outgoing_contrib view", vertices);

    // arbitrary numbers to test
    int max_iters = 1000;
    double epsilon = 0.001;

    //for (int n=0; n < vertices; n++)
    Kokkos::parallel_for("initiation loop", vertices, KOKKOS_LAMBDA(int n){
        outgoing_contrib(n) = init_score / out_degree(n);
    });

    for (int iter=0; iter < max_iters; iter++) {
        double error = 0;

        //#pragma omp parallel for reduction(+ : error)
        //for (int u=0; u < vertices; u++) {
        Kokkos::parallel_reduce("Outer loop", range_policy(0, vertices), KOKKOS_LAMBDA(int u, double& reduction_error){
    

	        int v = in_neigh_indexPointer(u);
            int limit = in_neigh_indexPointer(u+1);

            float incoming_total = 0;

            for (; v < limit; v++){
		        int v_element = in_neigh_indices(v);    
            	incoming_total += outgoing_contrib(v_element);
            }

            float old_score = scores(u);
       	    scores(u) = base_score + kDamp * incoming_total;
            reduction_error += abs(scores(u) - old_score);
            outgoing_contrib(u) = scores(u) / out_degree(u);

    	}, error);

    	printf(" %2d    %lf\n", iter, error);
    	if (error < epsilon)
      	    break;
    }

    //return scores;
    

    for(int a = 0; a < vertices; a++){
    	cout << "score of " << a << " : " << scores(a) << endl;  
    }

    cout << " " << endl;

    /**
    // print top scores
    map<float,int> check;	

    for(int a = 0; a < scores.size(); a++){
        map[scores[a]] = a;
    }

    for(auto pair: check){
    	cout << pair.second << ": " << pair.first<< endl;
    }
    **/

    Kokkos::finalize();

    return 0;
}   