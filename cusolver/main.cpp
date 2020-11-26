#include <fmt/core.h>
#include <vector>
#include <cusolverDn.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <openacc.h>

// NB: Column-major ordering!!!
static void print_matrix(std::vector<double> M, std::pair<int,int> shape) {

  for(int ii=0; ii < shape.first; ++ii) {
    fmt::print("|");
    for(int jj=0; jj < shape.second; ++jj) {
      fmt::print(" {}", M[jj*shape.first + ii]);
    }
    fmt::print(" |\n");
  }
  fmt::print("\n");

}

int main(int argc, char *argv[])
{

  int m = 3;
  int n = 2;
  int ldM = m;
  std::vector<double>M = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
  print_matrix(M, std::make_pair(m, n));

  cusolverDnHandle_t cusolverH = NULL;
  cusolverStatus_t cusolver_status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

  // Ensure we use the same stream as OpenACC (not sure if this is necessary
  // here, but it is for Thrust calls).
  cusolverDnSetStream(cusolverH, (cudaStream_t)acc_get_cuda_stream(acc_async_sync));

  int lwork;
  cusolver_status = cusolverDnDgesvd_bufferSize(cusolverH, m, n, &lwork);
  assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);

  int ldS = std::min(m, n);
  std::vector<double> S(ldS, 0);
  int ldU = std::max(1, m);
  std::vector<double> U(ldU*m, 0);
  int ldVT = std::max(1, n);
  std::vector<double> VT(ldVT*n, 0);
  int devInfo;
    
  double *M_ = M.data();
  double *S_ = S.data();
  double *U_ = U.data();
  double *VT_ = VT.data();

#pragma acc data copyin(M_) copyout(S_[0:m], U_[0:m*m], VT_[0:n*n], devInfo[0:1])
  {
    double* work_ = (double *)acc_malloc(lwork * sizeof(double));
    double* rwork_ = (double *)acc_malloc((m-1) * sizeof(double));

    signed char jobu = 'A'; // all m columns of U
    signed char jobvt = 'A'; // all n columns of VT

    #pragma acc host_data use_device(M_, S_, U_, VT_, devInfo)
    {
      /*********************************
      *  This only works if m>=n !!!  *
      *********************************/
      cusolver_status = cusolverDnDgesvd(cusolverH, jobu, jobvt, m, n, M_, m, S_, U_, m, VT_, n, work_, lwork, rwork_, &devInfo);
      cudaError_t cudaStat1 = cudaDeviceSynchronize();
      assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
      // fmt::print("err = {}\n", devInfo_[0]);
    }

    acc_free(rwork_);
    acc_free(work_);
  }

  print_matrix(U, std::make_pair(ldU, m));
  print_matrix(S, std::make_pair(ldS, 1));
  print_matrix(VT, std::make_pair(ldVT, n));

  cusolverDnDestroy(cusolverH);

  fmt::print("\n-----------------------------------\n\n");

  M = {1.0, 4.0, 2.0, 2.0, 5.0, 1.0};
  int ldx = std::max(n, m);
  std::vector<double>x_truth = {1.0, 0.5};
  int ldb = std::max(n, m);
  std::vector<double>b = {2.0, 6.5, 2.5};
  std::vector<double>x(x_truth.size());

  cusolver_status = cusolverDnCreate(&cusolverH);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

  // =======================================================
  // create the params and info structure for the expert interface
  cusolverDnIRSParams_t gels_irs_params;
  cusolverDnIRSParamsCreate( &gels_irs_params );
  cusolverDnIRSInfos_t gels_irs_infos;
  cusolverDnIRSInfosCreate( &gels_irs_infos );

  // Set the main and the low precision of the solver DSgels 
  // D is for double S for single precision thus 
  // main_precision is CUSOLVER_R_FP64, low_precision is CUSOLVER_R_FP32
  cusolverDnIRSParamsSetSolverPrecisions( gels_irs_params, CUSOLVER_R_64F, CUSOLVER_R_32F );
  // Set the refinement solver.
  cusolverDnIRSParamsSetRefinementSolver( gels_irs_params, CUSOLVER_IRS_REFINE_CLASSICAL );
  // Get work buffer size
  size_t lwork_bytes = 0;
  cusolver_status = cusolverDnIRSXgels_bufferSize(cusolverH, gels_irs_params, m, n, 1, &lwork_bytes);
  assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

  double* x_ = x.data();
  double* b_ = b.data();
  devInfo = 0;

#pragma acc data copyin(M_[m*n], b_[m]) copyout(x_[n]) copy(devInfo)
  {
    void* workspace_ = acc_malloc(lwork_bytes * sizeof(char));
    int niters = 0;

    #pragma acc host_data use_device(M_, b_, x_, devInfo)
    {
      // cusolver_status = cusolverDnDSgels(cusolverH, m, n, 1, M_, ldM, b_, ldb, x_, ldx, workspace_, lwork_bytes, &niters, &devInfo);
      cusolver_status = cusolverDnIRSXgels(cusolverH, gels_irs_params, gels_irs_infos, m, n, 1, (void *)M_, ldM, (void *)b_, ldb, (void *)x_, ldx, workspace_, lwork_bytes, &niters, &devInfo);
      cudaError_t cudaStat1 = cudaDeviceSynchronize();
      assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    }

    acc_free(workspace_);
  }

//   cusolver_status = cusolverDnCreate(&cusolverH);
//   assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
//   cusolverDnSetStream(cusolverH, (cudaStream_t)acc_get_cuda_stream(acc_async_sync));

//   size_t lwork_bytes = 0;
//   cusolver_status = cusolverDnDSgels_bufferSize(cusolverH, m, n, 1, NULL, ldM, NULL, ldb, NULL, ldx, NULL, &lwork_bytes);
//   assert(cusolver_status == CUSOLVER_STATUS_SUCCESS);
//   fmt::print("lwork_bytes = {}\n", lwork_bytes);

//   double* x_ = x.data();
//   double* b_ = b.data();
//   devInfo = 0;

// #pragma acc data copyin(M_[m*n], b_[3]) copyout(x_[ldx]) copy(devInfo)
//   {
//     void* workspace_ = acc_malloc(lwork_bytes * sizeof(char));
//     int niters = 0;

//     #pragma acc host_data use_device(M_, b_, x_, devInfo)
//     {
//       cusolver_status = cusolverDnDSgels(cusolverH, m, n, 1, M_, ldM, b_, ldb, x_, ldx, workspace_, lwork_bytes, &niters, &devInfo);
//       cudaError_t cudaStat1 = cudaDeviceSynchronize();
//     }

//     acc_free(workspace_);
//   }
  fmt::print("devInfo = {}\n", devInfo);

  print_matrix(M, std::make_pair(m, n));
  print_matrix(x, std::make_pair(1, n));
  print_matrix(b, std::make_pair(m, 1));

  cusolverDnDestroy(cusolverH);

  return 0;
}
