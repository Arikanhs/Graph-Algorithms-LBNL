#include <bits/stdc++.h>
#include <vector>
#include <iostream>
#include <list>
#include <algorithm>
#include <unordered_map>

#include <sstream>
#include <fstream>

#include <omp.h>

//#include <Kokkos_Core.hpp>

using namespace std;

int main(int argc, char* argv[]) {
    
    // Command line processing
    if(argc != 2){
        cout << "Usage ./executable <Input File>" << endl;
        exit(1);
    }

    string inputString = argv[1];




    // Kokkos::initialize();

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
                cout << "Failed to extract integers from line" << endl;
            }

        }
    }

    // Turning the data into CSR structure 

    vector<int> indexPointer;
    vector<int> indices; 

    // index for indexPointer vector
    int i = 0;

    // index for indices vector
    //int j = 0;

    int count = 0;

    //indexPointer[0] = 0;
    indexPointer.push_back(0);

    // Populate the CSR
    for(int a = 0; a < vertices; a++){

        // Populate the indexPointer vector
        int size = adj[a].size();
        i += size;
        //indexPointer[a+1] = i;
        indexPointer.push_back(i);

        // Populate the indices vector
        for(int b = 0; b < size; b++){
            indices.push_back(adj[a][b]);   
            //count++;
        }
    }


    /**
    for(int a = 0; a < adj.size(); a++){
       for(int b = 0; b < adj[a].size(); b++){
           cout << adj[a][b] << " ";
       }
       cout << "" <<endl;
    }
    **/


    /**
    for(int a = 0; a < indexPointer.size(); a++){
        printf("%d ", indexPointer[a]);
    }

    printf("\n"); 

    for(int a = 0; a < indices.size(); a++){
        printf("%d ", indices[a]);
    }
 
    printf("\n");
    **/



    //printf("section 2\n");

    vector<int> comp;
    comp.resize(vertices);

    #pragma omp parallel for
    for (int n = 0; n < vertices; n++) {
        //auto& n = *n_iter;
        comp[n] = n;
    }

    double time;

    time = omp_get_wtime();

    bool change = true;
    int num_iter = 0;

    while (change) {

        // HOOK SECTION
        change = false;
        num_iter++;

        #pragma omp parallel for
        for (int u = 0; u < vertices; u++) {

	    int v = indexPointer[u];	
            
	    int limit = indexPointer[u+1];

            for (v ; v < limit ; v++) {

                int v_element = indices[v];

                int comp_u = comp[u];
                int comp_v = comp[v_element];

                // if two nodes point to the same node go to the next iteration
                if (comp_u == comp_v) continue;

                // Hooking condition so lower component ID wins independent of direction
                // high comp is the node that has higher ID
                int high_comp = comp_u > comp_v ? comp_u : comp_v;
                // low comp is the node that has lower ID
                int low_comp = comp_u + (comp_v - high_comp);

                // if higher node points to itself point it to the lower comp
                if (high_comp == comp[high_comp]) {
                    change = true;
                    comp[high_comp] = low_comp;
                }
            }
        }

        // SHORTCUT SECTION
        #pragma omp parallel for
        for (int n = 0; n < vertices; n++) {
            // auto& n = *n_iter;
            while (comp[n] != comp[comp[n]]) {
                comp[n] = comp[comp[n]];
            }
        }
    }

    time = omp_get_wtime() - time;




    unordered_map<int,int> check;

    // validate the Shiloach-Vishkin algorithm 
    for(int n = 0; n < vertices; n++){
        int groupNo = comp[n];
        //if(comp[n])
        //printf("%lu "comp[n]);
	
        // cout << comp[n] << " " ;
        
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

    printf("The number of iterations Shiloach-Vishkin took:  %d \n", num_iter );
    printf("Total nodes SV iterated through:  %d \n", nodeCount );
    printf("Connected components count: %d \n", counter);
    printf("Time: %f \n", time);
    //Kokkos::finalize();

    return 0;
}

