#include "utils.hpp"

double stime(){
  struct timeval tv;
  struct timezone tz;
  gettimeofday(&tv, &tz);
  double mlsec = 1000.0 * ((double)tv.tv_sec + (double)tv.tv_usec/1000000.0);
  return mlsec/1000.0;
}

void show_params( uint n_runs, uint NP, uint n_evals, uint D, size_t PL, std::string PDB, std::string PS ){
  int NBA = (NP%32)? (NP/32)+1 : NP/32;
  int NTB = 32 * ceil((double) D / 32.0);
  int NTC = 32 * ceil((double) PL / 32.0);

  printf(" | PDB ID                                   %s\n", PDB.c_str());
  printf(" | Protein Sequence [%2zu]                   %s\n", PL, PS.c_str());
  printf(" | Number of Executions:                    %d\n", n_runs);
  printf(" | Population Size:                         %d\n", NP);
  printf(" | Number of Dimensions:                    %d\n", D);
  printf(" | Number of Function Evaluations:          %d\n", n_evals);
  printf(" | Optimization Function:                   3D-AB\n");
  printf(" +==============================================================+ \n");
  printf(" | Structure (A | S1)\n");
  printf(" | \t Number of Threads                        %d\n", 32);
  printf(" | \t Number of Blocks                         %d\n", NBA);
  printf(" | Structure (B | S2)\n");
  printf(" | \t Number of Threads                        %d\n", NTB);
  printf(" | \t Number of Blocks                         %d\n", NP);
  printf(" | Structure (C | S3)\n");
  printf(" | \t Number of Threads                        %d\n", NTC);
  printf(" | \t Number of Blocks                         %d\n", NP);
}
