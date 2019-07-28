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
  uint n_runs, NP, n_evals, hjeval;

  std::string PDB;

  try {
    po::options_description config("Opções");
    config.add_options()
      ("runs,r"    , po::value<uint>(&n_runs)->default_value(1)    , "Number of Executions" )
      ("pop_size,p", po::value<uint>(&NP)->default_value(20)       , "Population Size"      )
      ("protein,o", po::value<std::string>(&PDB)->default_value("1BXL"), "PDB ID"           )
      ("max_eval,e", po::value<uint>(&n_evals)->default_value(10e5), "Number of Function Evaluations")
      ("hj_eval,j", po::value<uint>(&hjeval)->default_value(10e5), "Number of Function Evaluations (LS)")
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

  thrust::device_vector<float>::iterator it_A;
  thrust::device_vector<float>::iterator it_B;

  float time  = 0.00;
  jDE * jde = new jDE(NP, n_dim, x_min, x_max);
  HookeJeeves * hj = new HookeJeeves(n_dim, PL, PS, 0.9, 1.0e-10);

  double hjres = 0;

  std::vector< std::pair<double, float> > stats;

  auto seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> random_i(0, NP-1);//[0, NP-1]
  std::uniform_int_distribution<int> random_d(0, n_dim-1); // [0, D-1]
  std::uniform_real_distribution<float> random(0, 1); // [0, 1)

  int Pb, Lb, C;
  if( n_dim < 45 ){
    Pb = 50;
    Lb = 10;
    C  = 5;
  } else {
    Pb = 25;
    Lb = 5;
    C  = 10;
  }

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

    it_A = thrust::min_element(thrust::device, d_fog.begin(), d_fog.end());
    b_id = thrust::distance(d_fog.begin(), it_A);

    float gb_e, lb_e;

    gb_e = lb_e = static_cast<float>(*it_A);

    //melhor encontrado em toda a execução
    thrust::host_vector<float> gb(n_dim);

    // melhor individuo na populaçao corrente
    thrust::host_vector<double> lb(n_dim);

    for( int nd = 0; nd < n_dim; nd++ ){
      gb[nd] = static_cast<float>(d_og[(b_id*n_dim) + nd]);
    }
    lb = gb;

    int g = 0;
    int r_c = 0;
    int local_reinit_counter = 0;
    int global_reinit_counter = 0;
    cudaEventRecord(start);
    for( uint evals = 0; evals < n_evals; ){
      // first update the F and CR for the first trial generation
      jde->update();

      // get the 3 mutually different indexs
      jde->index_gen();

      // get the best index
      it_A = thrust::min_element(thrust::device, d_fog.begin(), d_fog.end());
      b_id = thrust::distance(d_fog.begin(), it_A);

      // gen new trial vectors using BEST/1/BIN strategy
      jde->run_a(p_og, p_ng, b_id);

      B->compute(p_ng, p_fng);
      evals += NP;

      // Look if the trials has a new best solution
      it_B = thrust::min_element(thrust::device, d_fng.begin(), d_fng.end());

      if(lb_e - static_cast<float>(*it_B) > 0 ){
        lb_e = static_cast<float>(*it_B);
        r_c = 0;
      } else {
        r_c += NP;
      }

      // selection between current population and the new offsprings
      jde->selection_A(p_og, p_ng, p_fog, p_fng);

      if( r_c >= (Pb*n_dim) ){
        // printf(" RESET CHECK \n");
        if( local_reinit_counter >= 5){

          // update the global best
          it_A = thrust::min_element(thrust::device, d_fog.begin(), d_fog.end());
          b_id = thrust::distance(d_fog.begin(), it_A);

          if( static_cast<float>(*it_A) < gb_e ){
            gb_e = static_cast<float>(*it_A);
            for( int nd = 0; nd < n_dim; nd++ ){
              gb[nd] = static_cast<float>(d_og[(b_id*n_dim) + nd]);
            }
          }
          local_reinit_counter = 0;
          global_reinit_counter++;

          //printf(" | GLOBAL RESETING\n");
          thrust::transform(isb, isb + (n_dim * NP), d_og.begin(), prg(x_min, x_max, rd()));
          B->compute(p_og, p_fog);
          it_A = thrust::min_element(thrust::device, d_fog.begin(), d_fog.end());
          b_id = thrust::distance(d_fog.begin(), it_A);

          // new local best
          lb_e = static_cast<float>(*it_A);

          for( int nd = 0; nd < n_dim; nd++ ){
            lb[nd] = static_cast<float>(d_og[(b_id*n_dim) + nd]);
          }

        } else {
          //printf(" | LOCAL RESETING\n");

          // update the global best
          it_A = thrust::min_element(thrust::device, d_fog.begin(), d_fog.end());
          b_id = thrust::distance(d_fog.begin(), it_A);

          if( static_cast<float>(*it_A) < gb_e ){
            gb_e = static_cast<float>(*it_A);
            for( int nd = 0; nd < n_dim; nd++ ){
              gb[nd] = static_cast<float>(d_og[(b_id*n_dim) + nd]);
            }
          }

          for( int i = 0; i < NP; i++ ){
            for( int j = 0; j < n_dim; j++ ){
              d_og[(i*n_dim) + j] = gb[j];
            }
            for( int j = 0; j < C; j++ ){
              d_og[(i*n_dim) + random_d(rng)] = x_min + ((x_max - x_min) * random(rng));
            }
          }
          r_c = 0;
          local_reinit_counter++;

          B->compute(p_og, p_fog);
          it_A = thrust::min_element(thrust::device, d_fog.begin(), d_fog.end());
          b_id = thrust::distance(d_fog.begin(), it_A);

          // new local best
          lb_e = static_cast<float>(*it_A);

          for( int nd = 0; nd < n_dim; nd++ ){
            lb[nd] = static_cast<float>(d_og[(b_id*n_dim) + nd]);
          }
        }

      }

      g++;
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    /* End a Run */

    it_A = thrust::min_element(thrust::device, d_fog.begin(), d_fog.end());
    b_id = thrust::distance(d_fog.begin(), it_A);


    // printf(" %.5f vs %.5f\n",static_cast<float>(*it_A), gb_e );

    float ww;

    thrust::host_vector<double> H(n_dim);
    if( static_cast<float>(*it_A) < gb_e ){
      ww = static_cast<float>(*it_A);
      printf(" +==============================================================+ \n");
      printf(" | %-2d -- Promising region found with value: %8f.\n", run, ww);
      for( int nd = 0; nd < n_dim; nd++ ){
        H[nd] = static_cast<double>(d_og[(b_id * n_dim) + nd]);
      }
    } else {
      ww = gb_e;
      printf(" +==============================================================+ \n");
      printf(" | %-2d -- Promising region found with value: %8f.\n", run, ww);
      for( int nd = 0; nd < n_dim; nd++ )
        H[nd] = gb[nd];
    }
    printf(" | Number of local resets: %d\n", local_reinit_counter + (5 * global_reinit_counter));
    printf(" | Number of global resets: %d\n", global_reinit_counter);
    printf(" | \n");

    double tini, tend;
    tini = stime();
    hjres = hj->optimize(hjeval, H.data());
    tend = stime();

    printf(" | %-2d -- Conformation \n | ", run);
    for( int nd = 0; nd < n_dim; nd++ ){
      printf("%.5lf, ", (H[nd] * 180.0) / PI );
    }
    printf(" | \n\n");
    printf(" | Execution: %-2d Overall Best: %+.4f -> %+.4lf GPU Time (s): %.8f and HJ Time (s): %.8f\n", run, ww, hjres, time/1000.0, tend-tini);
    printf(" +==============================================================+ \n");

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
