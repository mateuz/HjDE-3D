/* CUDA includes */
#include "helper.cuh"

/* Local includes */
#include "utils.hpp"
#include "Benchmarks.cuh"
#include "jDE.cuh"
#include "3D_AB.cuh"
#include "HookeJeeves.hpp"

#define PI 3.1415926535897932384626433832795029

int main(int argc, char * argv[]){
  srand(time(NULL));
  uint n_runs, NP, n_evals;

  std::string PDB;

  try {
    po::options_description config("Opções");
    config.add_options()
      ("runs,r"    , po::value<uint>(&n_runs)->default_value(1)    , "Number of Executions" )
      ("pop_size,p", po::value<uint>(&NP)->default_value(20)       , "Population Size"      )
      ("protein,o", po::value<std::string>(&PDB)->default_value("1BXP"), "PDB ID"       )
      ("max_eval,e", po::value<uint>(&n_evals)->default_value(10e5), "Number of Function Evaluations")
      ("help,h", "Show help");

    po::options_description cmdline_options;
    cmdline_options.add(config);
    po::variables_map vm;
    store(po::command_line_parser(argc, argv).options(cmdline_options).run(), vm);
    notify(vm);
    if( vm.count("help") ){
      std::cout << cmdline_options << "\n";
      return 0;
    }
  } catch(std::exception& e) {
    std::cout << e.what() << "\n";
    return 1;
  }

  Benchmarks * B = NULL;
  B = new F3DAB(NP, PDB);

  uint PL = B->findSequence(PDB);
  std::string PS = B->getSequence(PDB);

  uint n_dim = (2 * PL) - 5;

  if( B == NULL ){
    printf("Error while instantiate the benchmark function on file Main.cu at line 49.\n");
    exit(EXIT_FAILURE);
  }

  float x_min = B->getMin();
  float x_max = B->getMax();

  printf(" +==============================================================+ \n");
  printf(" |                      EXECUTION PARAMETERS                    | \n");
  printf(" +==============================================================+ \n");
  show_params(n_runs, NP, n_evals, n_dim, PL, PDB, PS);
  printf(" +==============================================================+ \n");

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  thrust::device_vector<float> d_og(n_dim * NP);
  thrust::device_vector<float> d_ng(n_dim * NP);
  thrust::device_vector<float> d_bg(n_dim * NP);
  thrust::device_vector<float> d_fog(NP, 0.0);
  thrust::device_vector<float> d_fng(NP, 0.0);
  thrust::device_vector<float> d_fbg(NP, 0.0);
  thrust::device_vector<float> d_res(NP, 0.0);

  thrust::host_vector<float> h_og(n_dim * NP);
  thrust::host_vector<float> h_ng(n_dim * NP);
  thrust::host_vector<float> h_bg(n_dim * NP);
  thrust::host_vector<float> h_fog(NP);
  thrust::host_vector<float> h_fng(NP);
  thrust::host_vector<float> h_fbg(NP);

  float * p_og  = thrust::raw_pointer_cast(d_og.data());
  float * p_ng  = thrust::raw_pointer_cast(d_ng.data());
  float * p_bg  = thrust::raw_pointer_cast(d_bg.data());
  float * p_fog = thrust::raw_pointer_cast(d_fog.data());
  float * p_fng = thrust::raw_pointer_cast(d_fng.data());
  float * p_fbg = thrust::raw_pointer_cast(d_fbg.data());
  float * p_res = thrust::raw_pointer_cast(d_res.data());

  thrust::device_vector<float>::iterator it;

  float time  = 0.00;
  jDE * jde = new jDE(NP, n_dim, x_min, x_max);
  HookeJeeves * hj = new HookeJeeves(n_dim, PL, PS, 0.9, 1.0e-30);

  double hjres = 0;

  std::vector< std::pair<double, float> > stats;

  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> random_i(0, NP-1);//[0, NP-1]

  int b_id;
  for( uint run = 1; run <= n_runs; run++ ){
    // Randomly initiate the population

    // For practical use
    // random_device is generally only used to seed
    // a PRNG such as mt19937
    std::random_device rd;

    thrust::counting_iterator<uint> isb(0);
    thrust::transform(isb, isb + (n_dim * NP), d_og.begin(), prg(x_min, x_max, rd()));

    /* Starts a Run */

    //warm-up
    B->compute(p_og, p_fog);
    int g = 0;

    cudaEventRecord(start);
    for( uint evals = 0; evals < n_evals; ){
      // printf("> %d\n", g);
      g++;

      jde->index_gen();
      jde->run(p_og, p_ng);
      B->compute(p_ng, p_fng);
      evals += NP;

      // jde->selection(p_og, p_ng, p_fog, p_fng);
      // jde->update();

      // get the best index
      it   = thrust::min_element(thrust::device, d_fog.begin(), d_fog.end());
      b_id = thrust::distance(d_fog.begin(), it);

      jde->run_b(p_og, p_ng, p_bg, p_fog, p_fng, b_id);
      B->compute(p_bg, p_fbg);
      evals += NP;

      //selection between trial and best variations
      jde->selection(p_ng, p_bg, p_fng, p_fbg);

      //crowding between old generation and new trial vectors
      jde->crowding_selection(p_og, p_ng, p_fog, p_fng, p_res);

      jde->update();

      // if( g%1000 == 0 && g != 0 ){
      //   int b_idx = random_i(rng);
      //   // thrust::host_vector<float> H(d_og.begin() + (n_dim * b_idx), d_og.begin() + (n_dim * b_idx) + n_dim);
      //   thrust::host_vector<double> H(n_dim);
      //
      //   //device to host
      //   for( int d = 0; d < n_dim; d++ ){
      //     H[d] = static_cast<double>(d_og[(b_idx * n_dim) + d]);
      //     // printf("teste[%d] = %.15f;\n", d, (double)d_og[(b_idx * n_dim) + d]);
      //   }
      //
      //   // printf("Entring hooke jeeves with %.10lf\n", (float)d_fog[b_idx]);
      //   d_fog[b_idx] = static_cast<float>(hj->optimize(10000, H.data()));
      //
      //   //host to device
      //   for( int d = 0; d < n_dim; d++ ){
      //     d_og[(b_idx*n_dim) + d] = static_cast<float>(H[d]);
      //   }
      // }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    /* End a Run */

    it = thrust::min_element(thrust::device, d_fog.begin(), d_fog.end());
    // int d = thrust::distance(d_fog.begin(), it);

    float * iter = thrust::min_element(thrust::device, p_fog, p_fog + NP);
    int position = iter - p_fog;

    printf(" +==============================================================+ \n");
    // printf(" | Exploration phase finished.\n");
    printf(" | %-2d -- Promising region found with value: %8f.\n", run, static_cast<float>(*it));
    // printf(" +==============================================================+ \n");

    thrust::host_vector<double> H(n_dim);

    // //printf(" | R: ");
    for( int nd = 0; nd < n_dim; nd++ ){
      //   //printf("teste[%d] = %.20f;\n", nd, a);
      //   //printf("teste[%d] = %.30lf;\n", nd, static_cast<double>(a));
      H[nd] = static_cast<double>(d_og[(position * n_dim) + nd]);
    }

    // // printf("\n");
    double tini, tend;
    tini = stime();
    hjres = hj->optimize(1000, H.data());
    tend = stime();

    printf(" | %-2d -- Conformation \n | ", run);
    for( int nd = 0; nd < n_dim; nd++ ){
      printf("%.30lf, ", (H[nd] * 180.0) / PI );
      // printf("teste[%d] = %.30lf;\n", nd, H[nd]);
    }
    printf("\n");

    printf(" | Execution: %-2d Overall Best: %+.4f -> %+.4lf GPU Time (s): %.8f and HJ Time (s): %.8f\n", run, static_cast<float>(*it), hjres, time/1000.0, tend-tini);
    // printf(" | Execution: %-2d Overall Best: %+.4f GPU Time (ms): %.8f\n", run, static_cast<float>(*it), time);

    stats.push_back(std::make_pair(hjres, time));
    // stats.push_back(std::make_pair(static_cast<float>(*it), time));

    jde->reset();
  }

  /* Statistics of the Runs */
  float FO_mean = 0.0f, FO_std = 0.0f;
  float T_mean  = 0.0f, T_std  = 0.0f;
  for( auto it = stats.begin(); it != stats.end(); it++){
    FO_mean += it->first;
    T_mean  += it->second;
  }
  FO_mean /= n_runs;
  T_mean  /= n_runs;
  for( auto it = stats.begin(); it != stats.end(); it++){
    FO_std += (( it->first - FO_mean )*( it->first  - FO_mean ));
    T_std  += (( it->second - T_mean )*( it->second - T_mean  ));
  }
  FO_std /= n_runs;
  FO_std = sqrt(FO_std);
  T_std /= n_runs;
  T_std = sqrt(T_std);
  printf(" +==============================================================+ \n");
  printf(" |                     EXPERIMENTS RESULTS                      | \n");
  printf(" +==============================================================+ \n");
  printf(" | Objective Function:\n");
  printf(" | \t mean:         %+.20E\n", FO_mean);
  printf(" | \t std:          %+.20E\n", FO_std);
  printf(" | Execution Time (ms): \n");
  printf(" | \t mean:         %+.3lf\n", T_mean);
  printf(" | \t std:          %+.3lf\n", T_std);
  printf(" +==============================================================+ \n");

  return 0;
}
