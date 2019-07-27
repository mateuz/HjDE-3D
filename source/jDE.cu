#include "jDE.cuh"

jDE::jDE( uint _s, uint _ndim, float _x_min, float _x_max ):
  NP(_s),
  n_dim(_ndim),
  x_min(_x_min),
  x_max(_x_max)
{
  checkCudaErrors(cudaMalloc((void **)&F,  NP * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&CR, NP * sizeof(float)));
  thrust::fill(thrust::device, F , F  + NP, 0.50);
  thrust::fill(thrust::device, CR, CR + NP, 0.90);

  checkCudaErrors(cudaMalloc((void **)&T_F,  NP * sizeof(float)));
  checkCudaErrors(cudaMalloc((void **)&T_CR, NP * sizeof(float)));
  thrust::fill(thrust::device, T_F , T_F  + NP, 0.50);
  thrust::fill(thrust::device, T_CR, T_CR + NP, 0.90);

  Configuration conf;
  conf.x_min = x_min;
  conf.x_max = x_max;
  conf.ps = NP;
  conf.n_dim = n_dim;

  checkCudaErrors(cudaMemcpyToSymbol(params, &conf, sizeof(Configuration)));
  checkCudaErrors(cudaMalloc((void **)&rseq, NP * sizeof(uint)));
  checkCudaErrors(cudaMalloc((void **)&fseq, 3 * NP * sizeof(uint)));
  checkCudaErrors(cudaMalloc((void **)&d_states, NP * sizeof(curandStateXORWOW_t)));
  thrust::sequence(thrust::device, rseq, rseq + NP);

  NT_A.x = 32;
  NB_A.x = (NP%32)? (NP/32)+1 : NP/32;

  NT_B.x = 32 * ceil((double) n_dim / 32.0);
  NB_B.x = NP;

  std::random_device rd;
  unsigned int seed = rd();
  setup_kernel<<<NT_A, NB_A>>>(d_states, seed);
  checkCudaErrors(cudaGetLastError());

  checkCudaErrors(cudaMalloc((void **)&d_states2, NP * n_dim * sizeof(curandStateXORWOW_t)));
  setup_kernel2<<<NB_B, NT_B>>>(d_states2, seed);
  checkCudaErrors(cudaGetLastError());
}

jDE::~jDE()
{
  checkCudaErrors(cudaFree(F));
  checkCudaErrors(cudaFree(CR));
  checkCudaErrors(cudaFree(T_F));
  checkCudaErrors(cudaFree(T_CR));
  checkCudaErrors(cudaFree(rseq));
  checkCudaErrors(cudaFree(fseq));
  checkCudaErrors(cudaFree(d_states));
  checkCudaErrors(cudaFree(d_states2));
}

void jDE::reset(){
  thrust::fill(thrust::device, F , F  + NP, 0.50);
  thrust::fill(thrust::device, CR, CR + NP, 0.90);

  thrust::fill(thrust::device, T_F , T_F  + NP, 0.50);
  thrust::fill(thrust::device, T_CR, T_CR + NP, 0.90);
}

void jDE::update(){
  updateK<<<NB_A, NT_A>>>(d_states, F, CR, T_F, T_CR);
  checkCudaErrors(cudaGetLastError());
}

/*
 * fog == fitness of the old offspring
 * fng == fitness of the new offspring
 * BI  == best individual index
 */
void jDE::run_a(float * og, float * ng, uint BI){
  best_DE_01<<<NB_B, NT_B>>>(d_states2, og, ng, T_F, T_CR, fseq, BI);
  checkCudaErrors(cudaGetLastError());
}

__global__ void best_DE_01(curandState * rng, float * og, float * ng, float * F, float * CR, uint * fseq, uint pbest){
  uint id_d, id_p, ps, n_dim;

  id_d = blockIdx.x;
  id_p = threadIdx.x;

  n_dim = params.n_dim;
  ps = params.ps;

  __syncthreads();

  if( id_p < n_dim ){
    curandState random = rng[ id_d * id_p ];

    __shared__ uint n1, n2, p1, p2, p3, rnbr, pb;
    __shared__ float mF, mCR, ub, lb;

    if( id_p == 0 ){
      lb = params.x_min;
      ub = params.x_max;

      n1 = fseq[id_d];
      n2 = fseq[id_d + ps];

      mF  = F[id_d];
      mCR = CR[id_d];

      p1 = id_d * n_dim;
      p2 = n2 * n_dim;
      p3 = n1 * n_dim;
      pb = pbest * n_dim;

      rnbr = curand(&random) % n_dim;
    }

    __syncthreads();

    if( curand_uniform(&random) <= mCR || (id_p == rnbr) ){
      float T  = og[pb + id_p] + mF * (og[p2 + id_p] - og[p3 + id_p]);

      // check bounds
      if( T < lb ){
        T = ub + T + ub;
      } else if( T > ub ){
        T = lb + T + lb;
      }

      ng[p1 + id_p] = T;
    } else {
      ng[p1 + id_p] = og[p1 + id_p];
    }
    rng[id_d * id_p ] = random;
  }
}

void jDE::run_b(float * og, float * ng, float * bg, float * fog, float * fng, uint b_id){
  best_DE_02<<<NB_B, NT_B>>>(og, ng, bg, fog, fng, b_id);
  checkCudaErrors(cudaGetLastError());
}

__global__ void best_DE_02(float * og, float * ng, float * bnew, float * fog, float * fng, uint pbest){
  uint id_d, id_p, n_dim;

  //id_g = threadIdx.x + blockDim.x * blockIdx.x;

  id_d  = blockIdx.x;
	id_p  = threadIdx.x;

  n_dim = params.n_dim;

  if( id_p < n_dim ){
    __shared__ uint p1;
    __shared__ uint pb;

    __shared__ float _FA;
    __shared__ float _FB;

    __syncthreads();

    if( id_p == 0 ){
      p1 = id_d * n_dim;
      pb = pbest * n_dim;

      _FA = fog[id_d];
      _FB = fng[id_d];
    }

    __syncthreads();

    // if(id_p == 0 && id_d == 0){
    //   for(int i = 0; i < n_dim; i++){
    //     printf("teste[%d] = %.3f;\n", i, og[pb + i]);
    //   }
    // }

    if( _FB <= _FA ){
      bnew[p1 + id_p] = og[pb + id_p] + 0.5 * (ng[p1 + id_p] - og[p1 + id_p]);

      //check bounds
      if( bnew[p1 + id_p] <= params.x_min ){
        bnew[p1 + id_p] += 2.0 * params.x_max;
      } else if( bnew[p1 + id_p] > params.x_max ){
        bnew[p1 + id_p] += 2.0 * params.x_min;
      }
    } else {
      bnew[p1 + id_p] = ng[p1 + id_p];
    }
  }
}

void jDE::index_gen(){
  iGen<<<NB_A, NT_A>>>(d_states, rseq, fseq);
  checkCudaErrors(cudaGetLastError());
}

void jDE::selection_A(float * og, float * ng, float * fog, float * fng){
  selectionK<<<NB_A, NT_A>>>(og, ng, fog, fng, F, CR, T_F, T_CR);
  checkCudaErrors(cudaGetLastError());
}

void jDE::selection_B(float * og, float * ng, float * fog, float * fng){
  selectionK2<<<NB_A, NT_A>>>(og, ng, fog, fng, T_F, T_CR);
  checkCudaErrors(cudaGetLastError());
}

/*
 * Update F and CR values accordly with jDE algorithm.
 *
 * F_Lower, F_Upper and T are constant variables declared
 * on constants header
 */
__global__ void updateK(curandState * g_state, float * d_F, float * d_CR, float * d_TF, float * d_TCR) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;

  uint ps = params.ps;

  if( index < ps ){
    curandState localState;
    localState = g_state[index];

    //(0, 1]
    float r1, r2, r3, r4;
    r1 = curand_uniform(&localState);
    r2 = curand_uniform(&localState);
    r3 = curand_uniform(&localState);
    r4 = curand_uniform(&localState);

    if (r2 < T){
      d_TF[index] = F_Lower + (r1 * F_Upper);
    } else {
      d_TF[index] = d_F[index];
    }

    if (r4 < T){
      d_TCR[index] = r3;
    } else {
      d_TCR[index] = d_CR[index];
    }

    g_state[index] = localState;
  }
}

/*
 * Performs the selection step
 * In this case, each thread is a individual
 * og -> Old genes, the previous generation offspring
 * ng -> New genes, the new generation offsprings
 * fog -> fitness of the old offspring
 * fng -> fitness of the new offspring
 */
__global__ void selectionK(float * og, float * ng, float * fog, float * fng, float * d_F, float * d_CR, float * d_TF, float * d_TCR){
  uint index = threadIdx.x + blockDim.x * blockIdx.x;
  uint ps = params.ps;

  if( index < ps ){
    uint ndim = params.n_dim;
    if( fng[index] <= fog[index] ){
      memcpy(og + (ndim * index), ng + (ndim * index), ndim * sizeof(float));
      fog[index] = fng[index];
      d_F[index] = d_TF[index];
      d_CR[index] = d_TCR[index];
   }
  }
}

__global__ void selectionK2(float * ng, float * bg, float * fng, float * fbg, float * d_TF, float * d_TCR){
  uint index = threadIdx.x + blockDim.x * blockIdx.x;
  uint ps = params.ps;

  if( index < ps ){
    uint ndim = params.n_dim;
    if( fbg[index] < fng[index] ){
      memcpy(ng + (ndim * index), bg + (ndim * index), ndim * sizeof(float));
      fng[index]   = fbg[index];
      d_TF[index]  = 0.5;
      d_TCR[index] = 0.9;
   }
  }
}


/*
 * Generate 3 different indexs to DE/rand/1/bin.
 * @TODO:
 *  + rseq on constant memory;
 */
__global__ void iGen(curandState * g_state, uint * rseq, uint * fseq){
  uint index = threadIdx.x + blockDim.x * blockIdx.x;

  uint ps = params.ps;
  if( index < ps ){
    curandState s = g_state[index];

    uint n1, n2, n3;

    n1 = curand(&s) % ps;
    if( rseq[n1] == index )
      n1 = (n1 + 1) % ps;

    n2 = ( curand(&s) % ((int)ps/3) ) + 1;
    if( rseq[(n1 + n2) % ps] == index )
      n2 = (n2 + 1) % ps;

    n3 = ( curand(&s) % ((int)ps/3) ) + 1;
    if( rseq[(n1 + n2 + n3) % ps] == index )
      n3 = (n3 + 1 ) % ps;

    fseq[index] = rseq[n1];
    fseq[index+ps] = rseq[(n1+n2)%ps];
    fseq[index+ps+ps] = rseq[(n1+n2+n3)%ps];

    g_state[index] = s;
    //printf("[%-3d] %-3d | %-3d | %-3d\n", index, rseq[n1], rseq[(n1+n2)%ps], rseq[(n1+n2+n3)%ps]);
  }
}

/* Each thread gets same seed, a different sequence number, no offset */
__global__ void setup_kernel(curandState * random, uint seed){
  uint index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index < params.ps)
    curand_init(seed, index, 0, &random[index]);
}

/*
 *
 * Setup kernel version 2
 *
 */
__global__ void setup_kernel2(curandState * random, uint seed){
  uint index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index < params.ps * params.n_dim)
    curand_init(seed, index, 0, &random[index]);
}
