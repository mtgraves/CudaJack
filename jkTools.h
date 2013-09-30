/**
 * @file jackknifeTools.h
 * @author Max Graves
 * @date 28-SEP-2013
 *
 * @brief Define Class for tools needed to perform jackknife analysis of data.
 */

#include <stdio.h>
#include <stdlib.h>
#include <glob.h>
#include <math.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

#ifndef JACKKNIFETOOLS_H
#define JACKKNIFETOOLS_H

typedef vector<double> dVec;
typedef vector<dVec> vec_dVec;

/******************************************************************************
 * JackKnifeTools Class.
 ******************************************************************************/
class JackKnifeTools{
    public:
        JackKnifeTools();
        ~JackKnifeTools();

        /* function to check if file is empty */
        bool IsEmpty(ifstream &);

        /* function to glob for string in CWD */
        vector<string> Glob(const string &);

        /* function to read in columns of data separated by whitespace */
        vec_dVec ReadData(ifstream &);

        /* function to perform jackknife analysis of correlated data */
        void JkAnalysis(const dVec &, const dVec &, const dVec &, dVec&);
};
#endif
