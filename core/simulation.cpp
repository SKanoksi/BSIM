/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  simulation.cpp (main):
    BSIM main

\*****************************************************************************/

#include "simulation.hpp"


BSIM_model::BSIM_model(const BSIM::Vec2<lpInt> domain_id) :
            domain(new BSIM::Parallel_Domain(domain_id)),
            domain_size(this->domain->BSIM_var[0]->val->size),
            domain_edge(this->domain->BSIM_var[0]->val->edge),
            dt(static_cast<Float>(this->timestep_second)/static_cast<Float>(this->timestep_divide_second))
{
  for(lpInt i=0 ; i != this->num_thread ; ++i)
  {
    this->omp_temp_array.push_back( new BSIM::CubicTemp<Float>(domain_size, domain_edge) );
  }
  for(lpInt i=0 ; i != this->num_thread ; ++i)
  {
    this->omp_temp_coeff.push_back( new Float[6] );
  }
  for(lpInt i=0 ; i != this->num_thread ; ++i)
  {
    this->omp_temp_workspace.push_back( new hpFloat[18] );
  }

}


BSIM_model::~BSIM_model()
{
  for(lpInt i=0 ; i != static_cast<lpInt>( this->omp_temp_array.size() ) ; ++i)
  {
    delete this->omp_temp_array[i] ;
  }
  for(lpInt i=0 ; i != static_cast<lpInt>( this->omp_temp_coeff.size() ) ; ++i)
  {
    delete this->omp_temp_coeff[i] ;
  }
  for(lpInt i=0 ; i != static_cast<lpInt>( this->omp_temp_workspace.size() ) ; ++i)
  {
    delete this->omp_temp_workspace[i] ;
  }

  delete domain ;
}


bool BSIM_model::forward_all()
{
  { // I. Dynamics

    // *** Source-Sink ***
    //forward_source(this->domain->BSIM_var[static_cast<lpInt>(BSIM_VAR::tag::T)], 0.001);

    // *** Growth-Decay ***
    //forward_growth(this->domain->BSIM_var[static_cast<lpInt>(BSIM_VAR::tag::T)], 0.001);

    // *** Both = Source-Sink + Growth-Decay ***
    //forward_source_growth(this->domain->BSIM_var[static_cast<lpInt>(BSIM_VAR::tag::T)], 0.001, -0.001);

    // Advection-Diffusion
    // dT/dt + u dT/dx + v dT/dy = DT/dt = K (d^2T/dx^2 + d^2T/dy^2)
    // --> As if separate them

/*
    // *** Advection ***
    // -- Along x axis --
    forward_advection_x(this->domain->BSIM_var[static_cast<lpInt>(BSIM_VAR::tag::T)],
                        this->domain->BSIM_var[static_cast<lpInt>(BSIM_VAR::tag::u)]);

    domain->exchange_NS_bc(static_cast<lpInt>(BSIM_VAR::tag::T));
    domain->wait_NS_bc(static_cast<lpInt>(BSIM_VAR::tag::T));

    // -- Along y axis --
    forward_advection_y(this->domain->BSIM_var[static_cast<lpInt>(BSIM_VAR::tag::T)],
                        this->domain->BSIM_var[static_cast<lpInt>(BSIM_VAR::tag::v)]);

    domain->exchange_EW_bc(static_cast<lpInt>(BSIM_VAR::tag::T));
    domain->wait_EW_bc(static_cast<lpInt>(BSIM_VAR::tag::T));


    // *** Diffusion ***
    const Float D = 1.0 ;
    const hpFloat Ddtx4 = 4*D*this->dt ;

    // -- Along x axis --
    forward_diffusion_x(this->domain->BSIM_var[static_cast<lpInt>(BSIM_VAR::tag::T)],
                        this->domain_edge.x,
                        Ddtx4);

    domain->exchange_NS_bc(static_cast<lpInt>(BSIM_VAR::tag::T));
    domain->wait_NS_bc(static_cast<lpInt>(BSIM_VAR::tag::T));

    // -- Along y axis --
    forward_diffusion_y(this->domain->BSIM_var[static_cast<lpInt>(BSIM_VAR::tag::T)],
                        this->domain_edge.y,
                        Ddtx4);

    domain->exchange_EW_bc(static_cast<lpInt>(BSIM_VAR::tag::T));
    domain->wait_EW_bc(static_cast<lpInt>(BSIM_VAR::tag::T));
*/

    // *** Burgers' ***
    const Float K = 1.0 ;

    // -- Along x axis --
    forward_burgers_x(this->domain->BSIM_var[static_cast<lpInt>(BSIM_VAR::tag::T)],
                      this->domain_edge.x,
                      K);

    domain->exchange_NS_bc(static_cast<lpInt>(BSIM_VAR::tag::T));
    domain->wait_NS_bc(static_cast<lpInt>(BSIM_VAR::tag::T));

    // -- Along y axis --
    forward_burgers_y(this->domain->BSIM_var[static_cast<lpInt>(BSIM_VAR::tag::T)],
                      this->domain_edge.y,
                      K);

    //domain->exchange_EW_bc(static_cast<lpInt>(BSIM_VAR::tag::T));
    //domain->wait_EW_bc(static_cast<lpInt>(BSIM_VAR::tag::T));

    domain->exchange_all_bc(static_cast<lpInt>(BSIM_VAR::tag::T));
    domain->wait_all_bc(static_cast<lpInt>(BSIM_VAR::tag::T));

  }

return true; }


void BSIM_model::forward_source(BSIM::CubicGrid2D *var, const Float source)
{
  const Float add_source = source * this->dt ;

  // --> val
  #pragma omp taskloop default(shared) firstprivate(add_source) mergeable
  for(lpInt j=0 ; j != this->domain_size.y ; ++j)
  {
    #pragma omp simd simdlen(SIMD_NUM_INST)
    for(lpInt i=0 ; i != (this->domain_size.x - this->domain_size.x%SIMD_NUM_INST) ; ++i)
    {
      var->val->at(i,j) += add_source ;
    }

    // Remaining
    if( this->domain_size.x%SIMD_NUM_INST ){
      for(lpInt i=(this->domain_size.x - this->domain_size.x%SIMD_NUM_INST) ; i != this->domain_size.x ; ++i)
      {
        var->val->at(i,j) += add_source ;
      }
    }
  }

return; }


void BSIM_model::forward_growth(BSIM::CubicGrid2D *var, const Float growth)
{
  const Float mul_growth = std::exp( growth*this->dt ) ;

  // --> val
  #pragma omp taskloop default(shared) firstprivate(mul_growth) mergeable
  for(lpInt j=0 ; j != this->domain_size.y ; ++j)
  {
    #pragma omp simd simdlen(SIMD_NUM_INST)
    for(lpInt i=0 ; i != (this->domain_size.x - this->domain_size.x%SIMD_NUM_INST) ; ++i)
    {
      var->val->at(i,j) *= mul_growth ;
    }

    // Remaining
    if( this->domain_size.x%SIMD_NUM_INST ){
      for(lpInt i=(this->domain_size.x - this->domain_size.x%SIMD_NUM_INST) ; i != this->domain_size.x ; ++i)
      {
        var->val->at(i,j) *= mul_growth ;
      }
    }
  }

return; }


void BSIM_model::forward_source_growth(BSIM::CubicGrid2D *var, const Float source, const Float growth)
{
  const Float mul_factor = static_cast<Float>(std::exp( static_cast<hpFloat>(growth*this->dt) )) ;
  const Float add_factor = source/growth * (mul_factor - 1.0) ;

  // --> val
  #pragma omp taskloop default(shared) firstprivate(mul_factor,add_factor) mergeable
  for(lpInt j=0 ; j != this->domain_size.y ; ++j)
  {
    #pragma omp simd simdlen(SIMD_NUM_INST)
    for(lpInt i=0 ; i != (this->domain_size.x - this->domain_size.x%SIMD_NUM_INST) ; ++i)
    {
      var->val->at(i,j) = var->val->at_sw(i,j)*mul_factor + add_factor ;
    }

    // Remaining
    if( this->domain_size.x%SIMD_NUM_INST ){
      for(lpInt i=(this->domain_size.x - this->domain_size.x%SIMD_NUM_INST) ; i != this->domain_size.x ; ++i)
      {
        var->val->at(i,j) = var->val->at_sw(i,j)*mul_factor + add_factor ;
      }
    }
  }

return; }


void BSIM_model::forward_advection_x(BSIM::CubicGrid2D *var, const BSIM::CubicGrid2D *u)
{
  const Float dx_var = var->spacing.x ;
  bool assumption = true ;

  #pragma omp taskloop default(shared) firstprivate(dx_var) mergeable
  for(lpInt j=0 ; j != this->domain_size.y ; ++j)
  {
    BSIM::CubicTemp<Float> *var_temp = this->omp_temp_array[omp_get_thread_num()] ;
    Float *temp = this->omp_temp_coeff[omp_get_thread_num()] ;

    // I. Compute result --> temp
    Float local_x_depart_pre  = - dx_var - u->val->at(-1,j)*this->dt ;
    Float local_x_depart      = - u->val->at(0,j)*this->dt ;
    for(lpInt i=0 ; i != this->domain_size.x ; ++i)
    {
      Float local_x_depart_next = dx_var*(i+1) - u->val->at(i+1,j)*this->dt ;

      Float dx_unequal = std::abs( local_x_depart_next - local_x_depart_pre )/2 ;
      if( (local_x_depart_pre - local_x_depart)*(local_x_depart_next - local_x_depart) > 0 ){
        assumption = false ;
      }

      // forwardX_advect
      const lpInt iL = var->floor_local_x_to_ix<lpInt>(local_x_depart);
      const Float l  = local_x_depart - iL*dx_var ;
      var->x_coeff(iL, j, temp);
      var_temp->val_x[i]  = BSIM::func::grid2d::compute_val(l, temp) * dx_unequal/dx_var ;
      var_temp->grad_x[i] = BSIM::func::grid2d::compute_grad(l, temp) * (1.0 - u->grad_x->at(i,j)*this->dt) * dx_unequal/dx_var ;

      // For next i
      local_x_depart_pre = local_x_depart ;
      local_x_depart     = local_x_depart_next ;
    }

    // II. Transfer result
    this->transfer_result_x(var, this->domain_size.x, j, var_temp);

  } // J LOOP

  if( !assumption ){
    std::cerr
    << "\nAN ASSUMPTION IS VIOLATED while running BSIM_MAIN::forward_advection_x \n"
    << "--> May have to rerun again with a smaller timestep_second." << std::endl;
  }

return; }


void BSIM_model::forward_advection_y(BSIM::CubicGrid2D *var, const BSIM::CubicGrid2D *v)
{
  const Float dy_var = var->spacing.y ;
  bool assumption = true ;

  #pragma omp taskloop default(shared) firstprivate(dy_var) mergeable
  for(lpInt i=0 ; i != this->domain_size.x ; ++i)
  {
    BSIM::CubicTemp<Float> *var_temp = this->omp_temp_array[omp_get_thread_num()] ;
    Float *temp = this->omp_temp_coeff[omp_get_thread_num()] ;

    // I. Compute result --> temp
    Float local_y_depart_pre  = - dy_var - v->val->at(i,-1)*this->dt ;
    Float local_y_depart      = - v->val->at(i,0)*this->dt ;
    for(lpInt j=0 ; j != this->domain_size.y ; ++j)
    {
      Float local_y_depart_next = dy_var*(j+1) - v->val->at(i,j+1)*this->dt ;

      Float dy_unequal = std::abs( local_y_depart_next - local_y_depart_pre )/2 ;
      if( (local_y_depart_pre - local_y_depart)*(local_y_depart_next - local_y_depart) > 0 ){
        assumption = false ;
      }

      // forwardY_advect
      const lpInt jL = var->floor_local_y_to_jy<lpInt>(local_y_depart);
      const Float l  = local_y_depart - jL*dy_var ;
      var->y_coeff(i, jL, temp);
      var_temp->val_y[j]  = BSIM::func::grid2d::compute_val(l, temp) * dy_unequal/dy_var ;
      var_temp->grad_y[j] = BSIM::func::grid2d::compute_grad(l, temp) * (1.0 - v->grad_y->at(i,j)*this->dt) * dy_unequal/dy_var ;

      // For next j
      local_y_depart_pre = local_y_depart ;
      local_y_depart     = local_y_depart_next ;
    }

    // II. Transfer result
    this->transfer_result_y(var, i, this->domain_size.y, var_temp);

  } // I LOOP

  if( !assumption ){
    std::cerr
    << "\nAN ASSUMPTION IS VIOLATED while running BSIM_MAIN::forward_advection_y \n"
    << "--> May have to rerun again with a smaller timestep_second." << std::endl;
  }

return; }


void BSIM_model::forward_diffusion_x(BSIM::CubicGrid2D *var, const lpInt nl_sigma, const hpFloat Ddtx4)
{
  const Float dx_var = var->spacing.x ;

  #pragma omp taskloop default(shared) firstprivate(dx_var) mergeable
  for(lpInt j=0 ; j != this->domain_size.y ; ++j)
  {
    BSIM::CubicTemp<Float> *var_temp = this->omp_temp_array[omp_get_thread_num()] ;
    Float *temp = this->omp_temp_coeff[omp_get_thread_num()] ;

    // I. Compute result --> temp
    #pragma omp simd simdlen(SIMD_NUM_INST)
    for(lpInt i=0 ; i != (this->domain_size.x - this->domain_size.x%SIMD_NUM_INST) ; ++i)
    {
      var_temp->val_x[i]  = 0. ;
      var_temp->grad_x[i] = 0. ;
      for(lpInt ni=i-nl_sigma ; ni != i+nl_sigma ; ++ni)
      {
        var->x_coeff(ni, j, temp);
        BSIM::Vec2<Float> output = BSIM_DYNAMICS::forward_diffuse(Ddtx4, dx_var, (ni-i)*dx_var, temp);
        var_temp->val_x[i]  += output.x ;
        var_temp->grad_x[i] += output.y ;
      }
    }
    if( this->domain_size.x%SIMD_NUM_INST ){
      for(lpInt i=(this->domain_size.x - this->domain_size.x%SIMD_NUM_INST) ; i != this->domain_size.x ; ++i)
      {
        var_temp->val_x[i]  = 0. ;
        var_temp->grad_x[i] = 0. ;
        for(lpInt ni=i-nl_sigma ; ni != i+nl_sigma ; ++ni)
        {
          var->x_coeff(ni, j, temp);
          BSIM::Vec2<Float> output = BSIM_DYNAMICS::forward_diffuse(Ddtx4, dx_var, (ni-i)*dx_var, temp);
          var_temp->val_x[i]  += output.x ;
          var_temp->grad_x[i] += output.y ;
        }
      }
    }

    // II. Transfer result
    this->transfer_result_x(var, this->domain_size.x, j, var_temp);

  } // J LOOP

return; }


void BSIM_model::forward_diffusion_y(BSIM::CubicGrid2D *var, const lpInt nl_sigma, const hpFloat Ddtx4)
{
  const Float dy_var = var->spacing.y ;

  #pragma omp taskloop default(shared) firstprivate(dy_var) mergeable
  for(lpInt i=0 ; i != this->domain_size.x ; ++i)
  {
    BSIM::CubicTemp<Float> *var_temp = this->omp_temp_array[omp_get_thread_num()] ;
    Float *temp = this->omp_temp_coeff[omp_get_thread_num()] ;

    // I. Compute result --> temp
    #pragma omp simd simdlen(SIMD_NUM_INST)
    for(lpInt j=0 ; j != (this->domain_size.y - this->domain_size.y%SIMD_NUM_INST) ; ++j)
    {
      var_temp->val_y[j]  = 0. ;
      var_temp->grad_y[j] = 0. ;
      for(lpInt nj=j-nl_sigma ; nj != j+nl_sigma ; ++nj)
      {
        var->y_coeff(i, nj, temp);
        BSIM::Vec2<Float> output = BSIM_DYNAMICS::forward_diffuse(Ddtx4, dy_var, (nj-j)*dy_var, temp);
        var_temp->val_y[j]  += output.x ;
        var_temp->grad_y[j] += output.y ;
      }
    }
    if( this->domain_size.y%SIMD_NUM_INST ){
      for(lpInt j=(this->domain_size.y - this->domain_size.y%SIMD_NUM_INST) ; j != this->domain_size.y ; ++j)
      {
        var_temp->val_y[j]  = 0. ;
        var_temp->grad_y[j] = 0. ;
        for(lpInt nj=j-nl_sigma ; nj != j+nl_sigma ; ++nj)
        {
          var->y_coeff(i, nj, temp);
          BSIM::Vec2<Float> output = BSIM_DYNAMICS::forward_diffuse(Ddtx4, dy_var, (nj-j)*dy_var, temp);
          var_temp->val_y[j]  += output.x ;
          var_temp->grad_y[j] += output.y ;
        }
      }
    }

    // II. Transfer result
    this->transfer_result_y(var, i, this->domain_size.y, var_temp);

  } // I LOOP

return; }


void BSIM_model::forward_burgers_x(BSIM::CubicGrid2D *var, const lpInt nl_sigma, const Float K)
{
  const Float dx_var = var->spacing.x ;
  const hpFloat Kdt = K*this->dt ;

  #pragma omp taskloop default(shared) firstprivate(dx_var) mergeable
  for(lpInt j=0 ; j != this->domain_size.y ; ++j)
  {
    BSIM::CubicTemp<Float> *var_temp = this->omp_temp_array[omp_get_thread_num()] ;
    Float   *temp      = this->omp_temp_coeff[omp_get_thread_num()] ;
    hpFloat *workspace = this->omp_temp_workspace[omp_get_thread_num()] ;

    // I. Integrate
    hpFloat cumsum = 0. ; // To reduce round-off error when summing several small numbers
    var_temp->int_x[0] = 0. ;
    for(lpInt i = 1-nl_sigma ; i != this->domain_size.x + nl_sigma ; ++i)
    {
      var->x_coeff(i-1, j, temp);
      cumsum += static_cast<hpFloat>( BSIM::func::grid2d::compute_int(dx_var, temp) ) ;
      var_temp->int_x[i+nl_sigma] = static_cast<Float>( cumsum );
    }

    // II. Main calculation
    for(lpInt i=0 ; i != this->domain_size.x ; ++i)
    {
      hpFloat P   = 0. ;
      hpFloat Px  = 0. ;
      hpFloat Pxx = 0. ;
      for(lpInt ix=i-nl_sigma ; ix != i+nl_sigma ; ++ix)
      {
        BSIM::func::grid2d::quintic_coeff(var_temp->int_x[ix+nl_sigma], var_temp->int_x[ix+1+nl_sigma],
                                          var->val->at(ix,j),           var->val->at(ix+1,j),
                                          var->grad_x->at(ix,j),        var->grad_x->at(ix+1,j),
                                          dx_var, temp);
        // Transform to dimensionless space
        temp[0] /= K ;
        temp[1] /= K ;
        temp[2] /= K ;
        temp[3] /= K ;
        temp[4] /= K ;
        temp[5] /= K ;
        // Burgers' <-> Diffusion in Hopf-Cole space
        BSIM_DYNAMICS::diffuse_in_hopfcole_cancelcommon(4*Kdt, dx_var, (ix-i)*dx_var, temp, workspace);
        P   += workspace[0] ;
        Px  += workspace[1] ;
        Pxx += workspace[2] ;
      }
      // Back to regular space
      var_temp->val_x[i]  = static_cast<Float>( K*BSIM_DYNAMICS::func::inverse_hopfcole_val(P, Px, Kdt) );
      var_temp->grad_x[i] = static_cast<Float>( K*BSIM_DYNAMICS::func::inverse_hopfcole_grad(P, Px, Pxx, Kdt) );
    }

    // III. Transfer result
    this->transfer_result_x(var, this->domain_size.x, j, var_temp);

  } // J LOOP

return; }


void BSIM_model::forward_burgers_y(BSIM::CubicGrid2D *var, const lpInt nl_sigma, const Float K)
{
  const Float dy_var = var->spacing.y ;
  const hpFloat Kdt = K*this->dt ;

  #pragma omp taskloop default(shared) firstprivate(dy_var) mergeable
  for(lpInt i=0 ; i != this->domain_size.x ; ++i)
  {
    BSIM::CubicTemp<Float> *var_temp = this->omp_temp_array[omp_get_thread_num()] ;
    Float *temp        = this->omp_temp_coeff[omp_get_thread_num()] ;
    hpFloat *workspace = this->omp_temp_workspace[omp_get_thread_num()] ;

    // I. Integrate
    hpFloat cumsum = 0. ; // To reduce round-off error when summing several small numbers
    var_temp->int_y[0] = 0. ;
    for(lpInt j = 1-nl_sigma ; j != this->domain_size.y + nl_sigma ; ++j)
    {
      var->y_coeff(i, j-1, temp);
      cumsum += static_cast<hpFloat>( BSIM::func::grid2d::compute_int(dy_var, temp) ) ;
      var_temp->int_y[j+nl_sigma] = static_cast<Float>( cumsum );
    }

    // II. Main calculation
    for(lpInt j=0 ; j != this->domain_size.y ; ++j)
    {
      hpFloat P   = 0. ;
      hpFloat Px  = 0. ;
      hpFloat Pxx = 0. ;
      for(lpInt jy=j-nl_sigma ; jy != j+nl_sigma ; ++jy)
      {
        BSIM::func::grid2d::quintic_coeff(var_temp->int_y[jy+nl_sigma], var_temp->int_y[jy+1+nl_sigma],
                                          var->val->at(i,jy),           var->val->at(i,jy+1),
                                          var->grad_y->at(i,jy),        var->grad_y->at(i,jy+1),
                                          dy_var, temp);
        // Transform to dimensionless space
        temp[0] /= K ;
        temp[1] /= K ;
        temp[2] /= K ;
        temp[3] /= K ;
        temp[4] /= K ;
        temp[5] /= K ;
        // Burgers' <-> Diffusion in Hopf-Cole space
        BSIM_DYNAMICS::diffuse_in_hopfcole_cancelcommon(4*Kdt, dy_var, (jy-j)*dy_var, temp, workspace);
        P   += workspace[0] ;
        Px  += workspace[1] ;
        Pxx += workspace[2] ;
      }
      // Back to regular space
      var_temp->val_y[j]  = static_cast<Float>( K*BSIM_DYNAMICS::func::inverse_hopfcole_val(P, Px, Kdt) );
      var_temp->grad_y[j] = static_cast<Float>( K*BSIM_DYNAMICS::func::inverse_hopfcole_grad(P, Px, Pxx, Kdt) );
    }

    // III. Transfer result
    this->transfer_result_y(var, i, this->domain_size.y, var_temp);

  } // I LOOP

return; }
