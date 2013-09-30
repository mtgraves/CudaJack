/**
 * @file jackknifeTools.cpp
 * @author Max Graves
 * @date 28-SEP-2013
 *
 * @brief Implementation file for jackknifeTools.
 */

#include "jkTools.h"

/*****************************************************************************
 * Constructor
 *****************************************************************************/
JackKnifeTools::JackKnifeTools (){
};

/*****************************************************************************
 * Destructor
 *****************************************************************************/
JackKnifeTools::~JackKnifeTools (){
};

/*****************************************************************************
 * Check if a file is empty.
 *****************************************************************************/
bool JackKnifeTools::IsEmpty(ifstream& pFile){
    return pFile.peek() == std::ifstream::traits_type::eof();
}

/*****************************************************************************
 * Glob for a given string in the current working directory.
 *****************************************************************************/
vector<string> JackKnifeTools::Glob(const string& pat){
    glob_t glob_result;
    glob(pat.c_str(),GLOB_TILDE,NULL,&glob_result);
    vector<string> ret;
    for(unsigned int i=0;i<glob_result.gl_pathc;++i){
        ret.push_back(string(glob_result.gl_pathv[i]));
    }
    globfree(&glob_result);
    return ret;
}

/*****************************************************************************
 * Read data from text file.  
 * Data must be in columns, separated by whitespace.
 *****************************************************************************/
vec_dVec JackKnifeTools::ReadData(ifstream& f) {
    string line;
    vec_dVec rows;
    while (getline(f, line)) {
        /* do not read in comment lines */
        if (line[0]!='#'){
            istringstream iss(line);
            rows.push_back(dVec());
            while(iss){
                string sub;
                double subD;
                iss >> sub;
                if (sub != ""){
                    subD = strtod(sub.c_str(),NULL);
                    rows.back().push_back(subD);
                } //endif
            } //endwhile
        } //endif
    } //endwhile
    return rows;
}

/*****************************************************************************
 * Return Jackknife average and standard error.
 *****************************************************************************/
void JackKnifeTools::JkAnalysis(const dVec& Cv1, const dVec& Cv2, const dVec& Cv3,
        dVec& jkResults) {
    
    int numBins = Cv1.size();
    
    double jkTerm1, jkTerm2, jkTerm3, jkTermTot;
    double jkAve=0.0, jkAveSq=0.0;  //jackknife averages
    double rawAve1=0.0, rawAve2=0.0, rawAve3=0.0;    // raw data averages


    vector<double> tempJKvec1(numBins),tempJKvec2(numBins),tempJKvec3(numBins);

    /* compute jackknife averages */
    for (int i=0; i<numBins; i++){
        /* get rid of one term from each vector */
        tempJKvec1 = Cv1, tempJKvec2 = Cv2, tempJKvec3 = Cv3;
        tempJKvec1.erase(tempJKvec1.begin() + i);
        tempJKvec2.erase(tempJKvec2.begin() + i);
        tempJKvec3.erase(tempJKvec3.begin() + i);

        /* compute average of subset of data */
        jkTerm1 = 0.0, jkTerm2 = 0.0, jkTerm3 = 0.0, jkTermTot = 0.0;
        for (int j=0; j<numBins-1; j++){
            jkTerm1 += tempJKvec1[j]/(1.0*(numBins-1));
            jkTerm2 += tempJKvec2[j]/(1.0*(numBins-1));
            jkTerm3 += tempJKvec3[j]/(1.0*(numBins-1));
        }
        jkTermTot = jkTerm1 - jkTerm2*jkTerm2 - jkTerm3;

        rawAve1 += Cv1[i]/(1.0*numBins);
        rawAve2 += Cv2[i]/(1.0*numBins);
        rawAve3 += Cv3[i]/(1.0*numBins);

        /* update running average */
        jkAve += jkTermTot/(1.0*numBins);
        jkAveSq += jkTermTot*jkTermTot/(1.0*numBins);
    }

    double rawAve = rawAve1 - rawAve2*rawAve2 - rawAve3;

    double actAve = 1.0*numBins*rawAve - 1.0*(numBins-1)*jkAve;
    double var = jkAveSq - jkAve*jkAve;         // variance
    double err = sqrt(1.0*(numBins-1)*var);     // standard error

    /* update vector with results */
    jkResults.push_back(actAve);
    jkResults.push_back(err);

}
