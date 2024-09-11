/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  cubicgrid2d.hpp (header):
    2D cubic grid for numerical grid of BSIM

\*****************************************************************************/

#ifndef BSIM_CUBICGRID2D_HPP
#define BSIM_CUBICGRID2D_HPP

#include <array2d.hpp>

// Future:
// CubicGrid2d == Float only --> other datatype uses array2d
// 2D vector grid ==> 2 x CubicGrid2D

namespace BSIM
{

class CubicGrid2D
{

public:
  const Vec2<Float> pos0, spacing ; // make a copy before applying Vec2 operation
  Array2D<Float> *val = nullptr ;
  Array2D<Float> *grad_x = nullptr ;
  Array2D<Float> *grad_y = nullptr ;

  CubicGrid2D(const Vec2<lpInt> &num_grid_core, const Vec2<lpInt> &num_grid_edge,
              const Vec2<Float>  &origin_pos,   const Vec2<Float> &grid_spacing);
  ~CubicGrid2D();

  // Delete default copy and move operators
  CubicGrid2D(const CubicGrid2D&) = delete ;
  CubicGrid2D(CubicGrid2D&&) = delete ;
  CubicGrid2D& operator=(const CubicGrid2D&) = delete ;
  CubicGrid2D& operator=(CubicGrid2D&&) = delete;

  CubicGrid2D* copy();

  void cal_grad();
  void cal_grad_x();
  void cal_grad_y();

  void x_coeff(const lpInt i, const lpInt j, Float *coeff_out);
  void y_coeff(const lpInt i, const lpInt j, Float *coeff_out);

  inline Float pos_at_ix(const lpInt i){
    return this->pos0.x + this->spacing.x*i ;
  }
  inline Float pos_at_jy(const lpInt j){
    return this->pos0.y + this->spacing.y*j ;
  }

  template <typename T>
  inline T floor_local_x_to_ix(const Float local_x){
    return static_cast<T>( std::floor(local_x/this->spacing.x) ) ;
  }
  template <typename T>
  inline T floor_local_y_to_jy(const Float local_y){
    return static_cast<T>( std::floor(local_y/this->spacing.y) ) ;
  }

};


namespace func::grid2d
{

  // For repeated calculation:
  // No complete function --> If we provide, the program would be slower
  // Origin == local point (0,0)

  inline Float compute_val(const Float l, const Float *coeff){
    return coeff[0] + coeff[1]*l + coeff[2]*l*l + coeff[3]*l*l*l ;
  }
  inline Float compute_grad(const Float l, const Float *coeff){
    return coeff[1] + coeff[2]*2.0*l + coeff[3]*3.0*l*l ;
  }
  inline Float compute_curv(const Float l, const Float *coeff){
    return coeff[2]*2.0 + coeff[3]*6.0*l ;
  }
  inline Float compute_int(const Float l, const Float *coeff){
    return coeff[0]*l + 0.5*coeff[1]*l*l + coeff[2]*l*l*l/3.0 + 0.25*coeff[3]*l*l*l*l ;
  }

  void cubic_coeff(const Float L, const Float R,
                   const Float grad_L, const Float grad_R,
                   const Float dx, Float *coeff_out);
  void quintic_coeff(const Float L, const Float R,
                     const Float grad_L, const Float grad_R,
                     const Float curv_L, const Float curv_R,
                     const Float dx, Float *coeff_out);

} // NAMESPACE BSIM::func::grid2d

} // NAMESPACE BSIM


#endif // BSIM_CUBICGRID2D_HPP
