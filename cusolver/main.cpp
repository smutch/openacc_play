#include <fmt/core.h>
#include <vector>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <openacc.h>

static void print_matrix(std::vector<double> M, std::pair<int,int> shape) {

  for(int ii=0; ii < shape.first; ++ii) {
    fmt::print("|");
    for(int jj=0; jj < shape.second; ++jj) {
      fmt::print(" {}", M[ii*shape.second + jj]);
    }
    fmt::print(" |\n");
  }
  fmt::print("\n");

}

int main(int argc, char *argv[])
{

  // wikipedia example
  auto m = 4;
  auto n = 5;
  std::vector<double>M = {1,0,0,0,2,0,0,3,0,0,0,0,0,0,0,0,2,0,0,0};
  print_matrix(M, std::make_pair(m, n));

  cusolverDnHandle_t cusolverH = NULL;
  cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

  cublasHandle_t cublasH = NULL;
  cublasStatus_t cublas_status = cublasCreate(&cublasH);
  assert(CUBLAS_STATUS_SUCCESS == cublas_status); 

  int lwork;
  cusolver_status = cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  std::vector<double> S(n, 0);
  std::vector<double> U(m*m, 0);
  std::vector<double> VT(n*n, 0);
  std::vector<int> devInfo(1, 0);
    
#pragma acc data copyin(M) copyout(S[n], U[m*m], VT[n*n], devInfo[1])
  {
    double* work = (double *)acc_malloc(lwork * sizeof(double));

    signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT
    cusolver_status = cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, M.data(), m, S.data(), U.data(), m, VT.data(), n, work, lwork, NULL, devInfo.data());
    cudaError_t cudaStat1 = cudaDeviceSynchronize();

    acc_free(work);
  }

  cublasDestroy(cublasH);
  cusolverDnDestroy(cusolverH);
  
  print_matrix(U, std::make_pair(m, m));
  return 0;
}
