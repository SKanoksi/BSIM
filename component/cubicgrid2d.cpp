/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  cubicgrid2d.cpp (main):
    2D cubic grid for numerical grid of BSIM

\*****************************************************************************/

#include "cubicgrid2d.hpp"

namespace BSIM
{

CubicGrid2D::CubicGrid2D(const Vec2<lpInt> &num_grid_core, const Vec2<lpInt> &num_grid_edge,
                         const Vec2<Float>  &origin_pos,   const Vec2<Float> &grid_spacing)
                         : pos0(origin_pos), spacing( func::vec2::abs(grid_spacing) )
{
  this->val    = new Array2D<Float>(num_grid_core, num_grid_edge);
  this->grad_x = new Array2D<Float>(num_grid_core, num_grid_edge);
  this->grad_y = new Array2D<Float>(num_grid_core, num_grid_edge);
}


CubicGrid2D::~CubicGrid2D()
{
  delete val ;
  delete grad_x ;
  delete grad_y ;
}


CubicGrid2D* CubicGrid2D::copy()
{
  CubicGrid2D *x = new CubicGrid2D(this->val->size, this->val->edge, this->pos0, this->spacing);

  x->val->copy_whole_from(*(this->val));
  x->grad_x->copy_whole_from(*(this->grad_x));
  x->grad_y->copy_whole_from(*(this->grad_y));

return x; }


void CubicGrid2D::cal_grad()
{
  this->cal_grad_x();
  this->cal_grad_y();

return; }


void CubicGrid2D::cal_grad_x()
{
  for(lpInt j=0 ; j != this->val->full_size.y ; ++j)
  {
    // West BC
    this->grad_x->at_sw(0,j) = ( this->val->at_sw(1,j) - this->val->at_sw(0,j) )/this->spacing.x ;

    // Loop expected to be vectorized
    lpInt nX = this->val->full_size.x - 2 ;
    Float dx = 2*this->spacing.x ;
    #pragma omp simd simdlen(SIMD_NUM_INST)
    for(lpInt i=1 ; i != 1+(nX - nX%SIMD_NUM_INST) ; ++i)
    {
      this->grad_x->at_sw(i,j) = ( this->val->at_sw(i+1,j) - this->val->at_sw(i-1,j) )/dx ;
    }

    // Remaining
    if( nX%SIMD_NUM_INST ){
      for(lpInt i=static_cast<lpInt>(nX + 1 - nX%SIMD_NUM_INST) ; i != 1+nX ; ++i)
      {
        this->grad_x->at_sw(i,j) = ( this->val->at_sw(i+1,j) - this->val->at_sw(i-1,j) )/dx ;
      }
    }

    // East BC
    this->grad_x->at_sw(nX+1,j) = ( this->val->at_sw(nX+1,j) - this->val->at_sw(nX,j) )/this->spacing.x ;
  }

return; }


void CubicGrid2D::cal_grad_y()
{
  // South BC
  {
    // Loop expected to be vectorized
    #pragma omp simd simdlen(SIMD_NUM_INST)
    for(lpInt i=0 ; i != (this->val->full_size.x - this->val->full_size.x%SIMD_NUM_INST) ; ++i)
    {
      this->grad_y->at_sw(i,0) = ( this->val->at_sw(i,1) - this->val->at_sw(i,0) )/this->spacing.y ;
    }

    // Remaining
    if( this->val->full_size.x%SIMD_NUM_INST ){
      for(lpInt i=(this->val->full_size.x - this->val->full_size.x%SIMD_NUM_INST) ; i != this->val->full_size.x ; ++i)
      {
        this->grad_y->at_sw(i,0) = ( this->val->at_sw(i,1) - this->val->at_sw(i,0) )/this->spacing.y ;
      }
    }
  }

  // Main
  for(lpInt j=1 ; j != this->val->full_size.y-1 ; ++j)
  {
    // Loop expected to be vectorized
    Float dy = 2*this->spacing.y ;
    #pragma omp simd simdlen(SIMD_NUM_INST)
    for(lpInt i=0 ; i != (this->val->full_size.x - this->val->full_size.x%SIMD_NUM_INST) ; ++i)
    {
      this->grad_y->at_sw(i,j) = ( this->val->at_sw(i,j+1) - this->val->at_sw(i,j-1) )/dy ;
    }

    // Remaining
    if( this->val->full_size.x%SIMD_NUM_INST ){
      for(lpInt i=(this->val->full_size.x - this->val->full_size.x%SIMD_NUM_INST) ; i != this->val->full_size.x ; ++i)
      {
        this->grad_y->at_sw(i,j) = ( this->val->at_sw(i,j+1) - this->val->at_sw(i,j-1) )/dy ;
      }
    }
  }

  // North BC
  {
    const lpInt j = this->val->full_size.y-1 ;
    // Loop expected to be vectorized
    #pragma omp simd simdlen(SIMD_NUM_INST)
    for(lpInt i=0 ; i != (this->val->full_size.x - this->val->full_size.x%SIMD_NUM_INST) ; ++i)
    {
      this->grad_y->at_sw(i,j) = ( this->val->at_sw(i,j) - this->val->at_sw(i,j-1) )/this->spacing.y ;
    }

    // Remaining
    if( this->val->full_size.x%SIMD_NUM_INST ){
      for(lpInt i=(this->val->full_size.x - this->val->full_size.x%SIMD_NUM_INST) ; i != this->val->full_size.x ; ++i)
      {
        this->grad_y->at_sw(i,j) = ( this->val->at_sw(i,j) - this->val->at_sw(i,j-1) )/this->spacing.y ;
      }
    }
  }

return; }


void CubicGrid2D::x_coeff(const lpInt i, const lpInt j, Float *coeff_out)
{
  Float L = this->val->at(i,j)   ;
  Float R = this->val->at(i+1,j) ;
  Float grad_L = this->grad_x->at(i,j)   ;
  Float grad_R = this->grad_x->at(i+1,j) ;

  func::grid2d::cubic_coeff(L, R, grad_L, grad_R, this->spacing.x, coeff_out);
}


void CubicGrid2D::y_coeff(const lpInt i, const lpInt j, Float *coeff_out)
{
  Float L = this->val->at(i,j)   ;
  Float R = this->val->at(i,j+1) ;
  Float grad_L = this->grad_y->at(i,j)   ;
  Float grad_R = this->grad_y->at(i,j+1) ;

  func::grid2d::cubic_coeff(L, R, grad_L, grad_R, this->spacing.y, coeff_out);
}


/* Binary search
lpInt CubicGrid2D::floor_to_index(Float x, Float x0, Float dx, lpInt nx)
{
  lpInt iL = 0    ;
  lpInt iR = nx-1 ;
  while( iL+1 != iR )
  {
    lpFloat m = std::floor(static_cast<lpFloat>((iL+iR)/2)) ;
    if( x < x0 + m*dx ){
      iR = static_cast<lpInt>(m) ;
    }else{
      iL = static_cast<lpInt>(m) ;
    }
  }

return iL; }
*/


namespace func::grid2d
{

  void cubic_coeff(const Float L, const Float R,
                   const Float grad_L, const Float grad_R,
                   const Float dx, Float *coeff_out)
  {
    coeff_out[0] =      L ;
    coeff_out[1] = grad_L ;
    coeff_out[2] = (  3*( (R-L)/dx - grad_L ) - ( grad_R-grad_L ) )/dx      ;
    coeff_out[3] = ( -2*( (R-L)/dx - grad_L ) + ( grad_R-grad_L ) )/(dx*dx) ;
  }


  void quintic_coeff(const Float L, const Float R,
                     const Float grad_L, const Float grad_R,
                     const Float curv_L, const Float curv_R,
                     const Float dx, Float *coeff_out)
  {
    coeff_out[0] = L ;
    coeff_out[1] = grad_L ;
    coeff_out[2] = 0.5*curv_L ;

    hpFloat A = R - (coeff_out[0] + coeff_out[1]*dx + coeff_out[2]*dx*dx) ;
    A /= (dx*dx*dx) ;
    hpFloat B = grad_R - grad_L - curv_L*dx ;
    B /= (dx*dx) ;
    hpFloat C = curv_R - curv_L ;
    C *= 0.5/dx ;

    coeff_out[3] = static_cast<Float>(  10.0*A - 4.0*B +     C          ) ;
    coeff_out[4] = static_cast<Float>((-15.0*A + 7.0*B - 2.0*C)/dx      ) ;
    coeff_out[5] = static_cast<Float>((  6.0*A - 3.0*B +     C)/(dx*dx) ) ;
  }


} // NAMESPACE BSIM::func::grid2d


} // NAMESPACE BSIM
