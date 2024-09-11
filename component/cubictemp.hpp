/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  cubictemp.hpp:
    1D temporary arrays accompanying CubicGrid2D

\*****************************************************************************/

#ifndef BSIM_CUBICTEMP_HPP
#define BSIM_CUBICTEMP_HPP

#include <datatype.hpp>
#include <vec2.hpp>

namespace BSIM
{

template <typename T> class CubicTemp
{

public:
  T  *int_x = nullptr ;
  T  *int_y = nullptr ;
  T  *val_x = nullptr ;
  T  *val_y = nullptr ;
  T *grad_x = nullptr ;
  T *grad_y = nullptr ;

  CubicTemp(Vec2<lpInt> size, Vec2<lpInt> edge)
  {
    this->int_x  = new T[size.x + 2*edge.x] ;
    this->int_y  = new T[size.y + 2*edge.y] ;
    this->val_x  = new T[size.x] ;
    this->val_y  = new T[size.y] ;
    this->grad_x = new T[size.x] ;
    this->grad_y = new T[size.y] ;
  }

  // Delete default copy and move operators
  CubicTemp(const CubicTemp&) = delete ;
  CubicTemp(CubicTemp&&) = delete ;
  CubicTemp& operator=(const CubicTemp&) = delete ;
  CubicTemp& operator=(CubicTemp&&) = delete;

  ~CubicTemp()
  {
    delete[] this->int_x  ;
    delete[] this->int_y  ;
    delete[] this->val_x  ;
    delete[] this->val_y  ;
    delete[] this->grad_x ;
    delete[] this->grad_y ;
  }

};

} // NAMESPACE BSIM

#endif // BSIM_CUBICTEMP_HPP
