/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  shared2d.hpp:
    2D edge of array datatype

\*****************************************************************************/

#ifndef BSIM_SHARED2D_HPP
#define BSIM_SHARED2D_HPP

#include <ostream>
#include <iostream>

#include <datatype.hpp>
#include <vec2.hpp>


namespace BSIM
{

template <typename T> class Shared2D
{

public:
  T *sw = nullptr ;
  T *ss = nullptr ;
  T *se = nullptr ;
  T *ee = nullptr ;
  T *ne = nullptr ;
  T *nn = nullptr ;
  T *nw = nullptr ;
  T *ww = nullptr ;

  Shared2D(Vec2<lpInt> size, Vec2<lpInt> edge)
  {
		this->sw = new T[edge.x * edge.y] ;
		this->ss = new T[size.x * edge.y] ;
		this->se = new T[edge.x * edge.y] ;
		this->ee = new T[edge.x * size.y] ;
		this->ne = new T[edge.x * edge.y] ;
		this->nn = new T[size.x * edge.y] ;
		this->nw = new T[edge.x * edge.y] ;
		this->ww = new T[edge.x * size.y] ;
  }

  // Delete default copy and move operators
  Shared2D(const Shared2D&) = delete ;
  Shared2D(Shared2D&&) = delete ;
  Shared2D& operator=(const Shared2D&) = delete ;
  Shared2D& operator=(Shared2D&&) = delete;

  ~Shared2D()
  {
    delete[] this->sw ;
    delete[] this->ss ;
    delete[] this->se ;
    delete[] this->ee ;
    delete[] this->ne ;
    delete[] this->nn ;
    delete[] this->nw ;
    delete[] this->ww ;
  }

};

} // NAMESPACE BSIM

#endif // BSIM_SHARED2D_HPP
