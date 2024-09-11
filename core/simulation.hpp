/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  simulation.hpp (header):
    BSIM main

\*****************************************************************************/

#ifndef SIMULATION_HPP
#define SIMULATION_HPP

#include <omp.h>
#include <cubictemp.hpp>
#include <paralleldomain.hpp>

#include <dynamics.hpp>


class BSIM_model : public BSIM::BSIM_root
{
public:
  BSIM_model(const BSIM::Vec2<lpInt> domain_id);
  ~BSIM_model();

  BSIM::Parallel_Domain *domain = nullptr ;

  // Common runtime constants
  const BSIM::Vec2<lpInt> domain_size, domain_edge ;
  const Float dt ;

  // Note:
  //  MPI   ==> Parallel_Domain --> Shared2D + MPI_Isend | MPI_Irecv
  // OpenMP ==> #pragma omp task (thread pool) --> each variable x j-direction == All at the same time
  //        ==> No nest loop for LX cache opts as each variable grid (for NWP) should be able to fit in them (Modern CPU)
  // SIMD   ==> #pragma omp simd --> Nested loop i inside j loop where i in Array2D.at(i,j) (x-axis)
  //
  // Line ptr to Array: first == longest (z dir?)
  // Temporary = line (of the longest direction)
  bool forward_all();

private:
  std::vector<BSIM::CubicTemp<Float>*> omp_temp_array ;
  std::vector<Float*> omp_temp_coeff ;
  std::vector<hpFloat*> omp_temp_workspace ;

  void transfer_result_x(BSIM::CubicGrid2D *var, const lpInt nx, const lpInt j, const BSIM::CubicTemp<Float> *var_temp)
  {
    #pragma omp simd simdlen(SIMD_NUM_INST)
    for(lpInt i=0 ; i != (nx - nx%SIMD_NUM_INST) ; ++i)
    {
      var->val->at(i,j)    = var_temp->val_x[i] ;
      var->grad_x->at(i,j) = var_temp->grad_x[i] ;
    }
    if( nx%SIMD_NUM_INST ){
      for(lpInt i=(nx - nx%SIMD_NUM_INST) ; i != nx ; ++i)
      {
        var->val->at(i,j)    = var_temp->val_x[i] ;
        var->grad_x->at(i,j) = var_temp->grad_x[i] ;
      }
    }

  return; }

  void transfer_result_y(BSIM::CubicGrid2D *var, const lpInt i, const lpInt ny, const BSIM::CubicTemp<Float> *var_temp)
  {
    #pragma omp simd simdlen(SIMD_NUM_INST)
    for(lpInt j=0 ; j != (ny - ny%SIMD_NUM_INST) ; ++j)
    {
      var->val->at(i,j)    = var_temp->val_y[j] ;
      var->grad_y->at(i,j) = var_temp->grad_y[j] ;
    }
    if( ny%SIMD_NUM_INST ){
      for(lpInt j=(ny - ny%SIMD_NUM_INST) ; j != ny ; ++j)
      {
        var->val->at(i,j)    = var_temp->val_y[j] ;
        var->grad_y->at(i,j) = var_temp->grad_y[j] ;
      }
    }

  return; }

protected:
  // 1 Function call : 1 term in 1 complete equation
  // Update each full/pair of terms before others in the same equation !!!

  void forward_source(BSIM::CubicGrid2D *var, const Float source);
  void forward_growth(BSIM::CubicGrid2D *var, const Float growth);
  void forward_source_growth(BSIM::CubicGrid2D *var, const Float source, const Float growth);

  void forward_advection_x(BSIM::CubicGrid2D *var, const BSIM::CubicGrid2D *u);
  void forward_advection_y(BSIM::CubicGrid2D *var, const BSIM::CubicGrid2D *v);

  void forward_diffusion_x(BSIM::CubicGrid2D *var, const lpInt nl_sigma, const hpFloat Ddtx4);
  void forward_diffusion_y(BSIM::CubicGrid2D *var, const lpInt nl_sigma, const hpFloat Ddtx4);

  void forward_burgers_x(BSIM::CubicGrid2D *var, const lpInt nl_sigma, const Float K);
  void forward_burgers_y(BSIM::CubicGrid2D *var, const lpInt nl_sigma, const Float K);

};

#endif // SIMULATION_HPP
