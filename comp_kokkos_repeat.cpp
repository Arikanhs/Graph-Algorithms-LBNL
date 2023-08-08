#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include <list>
#include <algorithm>
#include <unordered_map>

#include <sstream>
#include <fstream>

#include <Kokkos_Core.hpp>

using namespace std;

int main(int argc, char* argv[]) {
    
    // Command line processing
    if(argc != 2){
        //cout << "Usage ./executable <Input File>" << endl;
	printf("Usage ./executable <Input File> \n");
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
	printf("Failed to open the file \n");
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

                if(secondInt > firstInt){

                    maxNum = max(firstInt,secondInt);

                    if(maxNum >= vertices){
                        adj.resize(maxNum + 1);
                        vertices = maxNum + 1;
                    }
                    
                    adj[firstInt].push_back(secondInt);
                    adj[secondInt].push_back(firstInt);

                    edges++;
                }    

            }else{
		printf("Failed to extract integers from line \n");    
               // cout << "Failed to extract integers from line" << endl;
            }

        }
    }

    // Turning the data into CSR structure 

    #ifdef KOKKOS_ENABLE_CUDA
    #define MemSpace Kokkos::CudaSpace
    #endif
    #ifdef KOKKOS_ENABLE_HIP
    #define MemSpace Kokkos::Experimental::HIPSpace
    #endif
    #ifdef KOKKOS_ENABLE_OPENMPTARGET
    #define MemSpace Kokkos::OpenMPTargetSpace
    #endif

    #ifndef MemSpace
    #define MemSpace Kokkos::HostSpace
    #endif

    using ExecSpace = MemSpace::execution_space;
    //using range_policy = Kokkos::RangePolicy<ExecSpace>;
    using range_policy = Kokkos::RangePolicy< Kokkos::Schedule<Kokkos::Dynamic>, ExecSpace >;

    // Allocate y, x vectors and Matrix A on device.
    //typedef Kokkos::View<int*, Kokkos::LayoutLeft, MemSpace>   ViewVectorType;
    typedef Kokkos::View<int*, MemSpace>   ViewVectorType;
    typedef Kokkos::View<double*, MemSpace>   ViewVectorTypeDouble;

    ViewVectorType indexPointerView( "index Pointer view", vertices+1 );
    ViewVectorType indicesView( "indices view", edges*2 );

    // Create host mirrors of device views.
    //ViewVectorType::HostMirror h_indexPointerView = Kokkos::create_mirror_view( indexPointerView  );
    //ViewVectorType::HostMirror h_indicesView = Kokkos::create_mirror_view( indicesView );

    // index for indexPointer vector
    int i = 0;

    // index for indices vector
    //int j = 0;

    int count = 0;

    //indexPointer[0] = 0;
    indexPointerView(0) = 0;

    // Populate the CSR
    for(int a = 0; a < vertices; a++){

        // Populate the indexPointer vector
        int size = adj[a].size();
        i += size;
        //indexPointer[a+1] = i;
        indexPointerView(a+1) = i;

        // Populate the indices vector
        for(int b = 0; b < size; b++){
            indicesView(count) = adj[a][b];   
            count++;
        }
    }

    // prints for Correction
    
    /**

    for(int a = 0; a < adj.size(); a++){
        for(int b = 0; b < adj[a].size(); b++){
            cout << adj[a][b] << " ";
        }
        cout << "" <<endl;
    }
    
    for(int a =0; a < vertices + 1; a++){
        cout << indexPointerView(a) << " " ;
    }
    cout << " " << endl;

    for(int a =0; a < edges*2; a++){
        cout << indicesView(a)<< " " ;
    }
    cout << " " << endl;

    **/

    //printf("section 2\n");

    ViewVectorType compView( "comp", vertices );
    ViewVectorTypeDouble timings("timings", 10 );
    //ViewVectorType iterations_SV("iter", 10);

    ViewVectorType change( "change", 1 );


    for(int i = 0; i < 15; i++){

    	//#pragma omp parallel for
    	for (int n = 0; n < vertices; n++) {
            //auto& n = *n_iter;
            compView(n) = n;
    	}

    	change(0) = 1;

    	//bool change = true;
    	int num_iter = 0;

    	//bool* changeptr = &change;

    	double time;
   	Kokkos::Timer timer;

        // while(change)
        while (change(0) == 1) {

            // HOOK SECTION
            //change = false;
	    change(0) = 0;

            num_iter++;

        
            Kokkos::parallel_for("first loop", range_policy(0,vertices), KOKKOS_LAMBDA(int u){
            //for(int u = 0; u < vertices; u++){    

                int v = indexPointerView(u);

                int limit = indexPointerView(u+1);

                for (; v < limit; v++) {

                    int v_element = indicesView(v);

                    int comp_u = compView(u);
                    int comp_v = compView(v_element);

                    // if two nodes point to the same node go to the next iteration
                    if (comp_u == comp_v) continue;

                    // Hooking condition so lower component ID wins independent of direction
                    // high comp is the node that has higher ID
                    int high_comp = comp_u > comp_v ? comp_u : comp_v;
                    // low comp is the node that has lower ID
                    int low_comp = comp_u + (comp_v - high_comp);

                    // if higher node points to itself point it to the lower comp
                    if (high_comp == compView(high_comp)) {	
                        //change = true;
		        //Kokkos::atomic_store(changeptr, true);
		        change(0) = 1;
                        compView(high_comp) = low_comp;
                    }
                }
            });

            // SHORTCUT SECTION
            Kokkos::parallel_for("second loop", range_policy(0,vertices), KOKKOS_LAMBDA(int n){
            //for (int n = 0; n < vertices; n++) {
            
                while (compView(n) != compView(compView(n))) {
                    compView(n) = compView(compView(n));
                }
            });
    	}

    	time = timer.seconds();

    	if(i >= 5){
            timings(i-5)= time;
    	}    

    	printf("Timing for execution %d: %f \n", i, time);
	printf("The number of iterations Shiloach-Vishkin took:  %d \n", num_iter );
    }
 

    double avg = 0;

    for(int i = 0; i < 10; i++){
    	avg += timings(i);
    }	

    avg = avg/10;

    printf("Average timing: %f \n", avg);

    // Validation of the result

    unordered_map<int,int> check;

    // validate the Shiloach-Vishkin algorithm 
    for(int n = 0; n < vertices; n++){
        int groupNo = compView(n);
        //if(comp[n])
        //printf("%lu "comp[n]);
	
        //cout << compView(n) << " " ;
        
        check[groupNo]++;
    }

    // printf("\n");

    int counter = 0;
    int nodeCount =0;
    for(auto pair: check){
        counter++;
        nodeCount += pair.second;
        //printf("%lu "pair.second);
	
        //cout << pair.second << " " ;
    }

    // printf("\n");

    //print statements

    printf("Total nodes SV iterated through:  %d \n", nodeCount );
    printf("Connected components count: %d \n", counter);
    //printf("Time: %f \n", time);

    cout << " " << endl; 

    Kokkos::finalize();

    return 0;
}

