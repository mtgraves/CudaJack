#include <stdio.h>
#include <glob.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/remove.h>

#include "jkTools.h"

using namespace std;

/* ============================================================================
 * GPU functions
 * =========================================================================*/
/*__global__ void add(int *a, int *b, int *c){
    int tid = threadIdx.x;       //handle data at this index
    if (tid < N){
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void average(double *a, double *b, double *c){
    int tid = threadIdx.x;       //handle data at this index
    if (tid < N){
        c[tid] = a[tid] + b[tid];
    }
}*/

/* ============================================================================
 * Begin Main.
 * =========================================================================*/
int main(void){

    JackKnifeTools jk;
 
    /* check if data file is empty before writing */
    ifstream filestr("JackknifeData.dat");
    if(!jk.IsEmpty(filestr)){
        cout<<"\nError: Data file is not empty!\n"<<endl;
        return 0;
    }

    /* open file to write analyzed data to disk */
    FILE *file;
    file = fopen("JackknifeData.dat", "a+");
    fprintf(file,"#%9s\t%10s\t%10s\n", "T", "Cv", "CvErr");
    
    /* vector of estimator file names */
    vector<string> estFiles = jk.Glob("*estimator*");

    /* loop over all estimator files */
    for (int i = 0; i<estFiles.size(); i++){
        cout<<estFiles[i]<<endl;
        string temp = estFiles[i].substr(13,6);
        
        /* read in file */
        ifstream inFile;
        inFile.open(estFiles[i].c_str());

        /* grab whitespace delimited data */
        vec_dVec allData = jk.ReadData(inFile);
       
        /* bookkeeping */
        const int nBin = allData.size();
        
        /* store specific heat data */
        thrust::host_vector<double> Cv1(nBin), Cv2(nBin), Cv3(nBin);
        for (int j=0; j<nBin; j++){
            Cv1[j] = allData[j][11];
            Cv2[j] = allData[j][12];
            Cv3[j] = allData[j][13];
        }
        
        /* delete all unused data from memory */
        vector<dVec>().swap(allData);

        double jkTerm1, jkTerm2, jkTerm3, jkTermTot;
        double jkAve=0.0;       // jackknife average
        double jkAveSq=0.0;
        double rawAve1=0.0, rawAve2=0.0, rawAve3=0.0;

        thrust::host_vector<double> tempJKvec1(nBin);
        thrust::host_vector<double> tempJKvec2(nBin);
        thrust::host_vector<double> tempJKvec3(nBin);
        
        /* compute jackknife averages */
        for (int j=0; j<nBin; j++){
            /* get rid of one term from each vector */
            tempJKvec1 = Cv1, tempJKvec2 = Cv2, tempJKvec3 = Cv3;
            tempJKvec1.erase(tempJKvec1.begin() + j);
            tempJKvec2.erase(tempJKvec2.begin() + j);
            tempJKvec3.erase(tempJKvec3.begin() + j);
            
            /* compute average of subset of data */
            jkTerm1 = 0.0, jkTerm2 = 0.0, jkTerm3 = 0.0, jkTermTot = 0.0;
            for (int l=0; l<nBin-1; l++){
                jkTerm1 += tempJKvec1[l]/(1.0*(nBin-1));
                jkTerm2 += tempJKvec2[l]/(1.0*(nBin-1));
                jkTerm3 += tempJKvec3[l]/(1.0*(nBin-1));
            }
            jkTermTot = jkTerm1 - jkTerm2*jkTerm2 - jkTerm3;

            rawAve1 += Cv1[j]/(1.0*nBin);
            rawAve2 += Cv2[j]/(1.0*nBin);
            rawAve3 += Cv3[j]/(1.0*nBin);

            /* update running average */
            jkAve += jkTermTot/(1.0*nBin);
            jkAveSq += jkTermTot*jkTermTot/(1.0*nBin);
        }
        double rawAve = rawAve1 - rawAve2*rawAve2 - rawAve3;

        double actAve = 1.0*nBin*rawAve - 1.0*(nBin-1)*jkAve;
        double var = jkAveSq - jkAve*jkAve;
        double err = sqrt(1.0*(nBin-1)*var);

        cout<<"est = "<<actAve<<" +/- "<<err<<endl;

        /* temp stuff for writing */
        //double Cv = 0.0;
        //double CvErr = 0.0;

        /* create device arrays */
        //thrust::device_vector<double> Cv1_dev = Cv1;
        //thrust::device_vector<double> Cv2_dev = Cv1;
        //thrust::device_vector<double> Cv3_dev = Cv1;
        //thrust::device_vector<double> JkTerms_dev(nBin);
     
        /* allocate memory of GPU */
        //cudaMalloc((void**)&dev_Cv1, nBin*sizeof(double));
        //cudaMalloc((void**)&dev_b, sizeof(int));
        //cudaMalloc((void**)&dev_c, sizeof(int));
        
        // copy 'a' and 'b' to GPU
        // HANDLE_ERROR( cudaMemcpy(dev_a,a,N*sizeof(int),
        //            cudaMemcpyHostToDevice));
        //cudaMemcpy(dev_a,a,N*sizeof(int),cudaMemcpyHostToDevice);
        // HANDLE_ERROR( cudaMemcpy(dev_b,b,N*sizeof(int),
        //            cudaMemcpyHostToDevice));
        //cudaMemcpy(dev_b,b,N*sizeof(int), cudaMemcpyHostToDevice);
        
        //add<<<200,N/200>>>(dev_a,dev_b,dev_c);

        // copy array 'c' back from GPU to CPU
        // HANDLE_ERROR( cudaMemcpy(c,dev_c,N*sizeof(int),
        //            cudaMemcpyDeviceToHost));
        //cudaMemcpy(c,dev_c,N*sizeof(int), cudaMemcpyDeviceToHost);

        // free memory allocated on GPU
        //cudaFree(dev_a);
        //cudaFree(dev_b);
        //cudaFree(dev_c);

        //cout<<temp<<endl;
        fprintf(file,"%10s\t%10f\t%10f\n", temp.c_str(), actAve, err);
    }

    fclose(file);
    return 0;
}
