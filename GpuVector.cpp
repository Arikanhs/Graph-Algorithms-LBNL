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
 
 
// Driver code
int main(int argc, char* argv[])
{

    // Command line processing
    if(argc != 2){
        //cout << "Usage ./TriCount4.host <Input File>" << endl;
        exit(1);
    }

    string inputString = argv[1];



    Kokkos::initialize();

    vector<vector<int>> adj;
    int vertices;
    int edges = 0;

    // Get the core count to print at the end
    //int core_count = Kokkos::HostSpace::execution_space::concurrency();

    // Initialize the graph and add edges from the input file
    //Graph g;

    //g.adj.resize(1134890);
    adj.resize(1);
    vertices = 1;

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
        //cout << "Failed to open the file" << endl;
        exit(1);
    }

    printf("print 1");

    while (getline(inputFile, line))
    {
	// check the commented lines    
	if(line[0] != '%'){    
            istringstream iss(line);
            int firstInt, secondInt;

            if(iss >> firstInt >> secondInt){

                if(secondInt > firstInt){
            	    maxNum = max(firstInt,secondInt);
            	    minNum = min(firstInt,secondInt);

            	    if(maxNum > vertices){
                	adj.resize(maxNum);
                	vertices = maxNum;
            	    }
            	   
            	    adj[minNum].push_back(maxNum);
            	    //g.addEdgeNIP(minNum,maxNum);
            	    edges++;
	        }

            }else{
                //cout << "Failed to extract integers from line" << endl;
            }

	 }
    }
    vertices++;


    printf("print 2");



    // Sort the graph
    //double timeSort;

    //Kokkos::Timer sortTimer;
    /**
    //Kokkos::parallel_for("Sort loop", vertices, KOKKOS_LAMBDA(int a){
    for(int a = 0; a < vertices; a++){
        int size = adj[a].size();
        for(int b = 0; b < size; b++){
            sort(adj[a].begin(), adj[a].end());
        }
    }
    //});
    **/
    //timeSort = sortTimer.seconds();


    printf("print 3");
	



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
    using range_policy = Kokkos::RangePolicy<ExecSpace>;
    //using range_policy = Kokkos::RangePolicy< Kokkos::Schedule<Kokkos::Dynamic>, ExecSpace >;

    // Allocate y, x vectors and Matrix A on device.
    typedef Kokkos::View<int*, Kokkos::LayoutLeft, MemSpace>   ViewVectorType;

    ViewVectorType indexPointerView( "index Pointer view", vertices+1 );
    ViewVectorType indicesView( "indices view", edges );

    // Create host mirrors of device views.
    ViewVectorType::HostMirror h_indexPointerView = Kokkos::create_mirror_view( indexPointerView  );
    ViewVectorType::HostMirror h_indicesView = Kokkos::create_mirror_view( indicesView );

    // index for indexPointer vector
    int i = 0;

    // index for indices vector
    //int j = 0;

    int count = 0;

    //indexPointer[0] = 0;
    h_indexPointerView(0) = 0;

    // Populate the CSR
    for(int a = 0; a < vertices; a++){

        // Populate the indexPointer vector
        int size = adj[a].size();
        i += size;
        //indexPointer[a+1] = i;
        h_indexPointerView(a+1) = i;

        // Populate the indices vector
        for(int b = 0; b < size; b++){
            h_indicesView(count) = adj[a][b];   
            count++;
        }
    }


    printf("print 4");



    // Deep copy host views to device views.
    Kokkos::deep_copy( indexPointerView, h_indexPointerView );
    Kokkos::deep_copy( indicesView, h_indicesView );



    //typedef Kokkos::TeamPolicy< Kokkos::Schedule<Kokkos::Dynamic> >    team_policy;
    //typedef Kokkos::TeamPolicy< Kokkos::Schedule<Kokkos::Dynamic>>::member_type    member_type;

    typedef Kokkos::TeamPolicy<>    team_policy;
    typedef Kokkos::TeamPolicy<>::member_type    member_type;

    double timeCsrView;

    int result = 0;

    Kokkos::Timer timer;

    Kokkos::parallel_reduce("Outer Reduction", team_policy(vertices -1, Kokkos::AUTO, 8), KOKKOS_LAMBDA ( const member_type& team, int &outer_reduction) {
	
    	int inner_reduction_result = 0;
	
    	// get the index
    	int a = team.league_rank();

    	int i = indexPointerView(a);

    	int limit = indexPointerView(a + 1);

	//int iterationIndexes = limit - i;

    	Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, i, limit), [=] (int b, int& inner_reduction){
	
	    int neighbor = indicesView(b);

	    int j = indexPointerView(neighbor);

	    int limit2 = indexPointerView(neighbor +1);

	    int k = b;

	    while(k < limit && j < limit2){
                if(indicesView(k) > indicesView(j)){
            	    j++;
                }else if(indicesView(j) > indicesView(k)){
                    k++;
                }else{
                    j++;
                    k++;
                    inner_reduction++;
                }
            }


        }, inner_reduction_result);

	Kokkos::single(Kokkos::PerTeam(team), [&] (){
		outer_reduction += inner_reduction_result;		
	});

    }, result);		    
    
    timeCsrView = timer.seconds();


    //Kokkos::fence();
    printf("print 5\n");


    printf("Triangle Count (View): %d\n", result);
    printf("Time (View): %f\n", timeCsrView);	
    //printf("Time to sort: %d\n", timeSort);
    printf("Vertex count: %d\n", vertices);
    printf("Edge count: %d\n", edges);


    Kokkos::finalize();


    return 0;



}

