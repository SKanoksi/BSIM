/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  Vec2.hpp:
    2D numerical values for default datatypes

\*****************************************************************************/

#ifndef BSIM_VEC2_HPP
#define BSIM_VEC2_HPP

#include <ostream>
#include <cmath>
#include <algorithm>

namespace BSIM
{

template <typename T> class Vec2
{
public:
  T x, y ;

  // *** CONSTRUCTOR ***
  Vec2() : x(static_cast<T>(0)), y(static_cast<T>(0)) {}
  Vec2(const T a, const T b) : x(a), y(b) {}
  template <typename A, typename B>
  Vec2(const A a, const B b) : x(static_cast<T>(a)),  y(static_cast<T>(b)) {}
  template <typename U>
  Vec2(const Vec2<U> &v) : x(static_cast<T>(v.x)),  y(static_cast<T>(v.y)) {}

  constexpr Vec2(const Vec2& v) : x(v.x), y(v.y) {}

  // *** OSTREAM ***
  friend std::ostream &operator<< (std::ostream & os, const Vec2<T> &v){
    os << "(" << v.x << "," << v.y << ")" ;
  return os; }

  // *** OPERATOR [SELF] ***

  template <typename U>
  T &operator[] (const U i){
    if( i == 1 )
      return y ;
    else
      return x ;
  }

  inline Vec2<T> &operator= (const Vec2<T> &v){
    this->x = v.x ;
    this->y = v.y ;
  return *this ; }

  template <typename U>
  inline Vec2<T> &operator= (const Vec2<U> &v){
    this->x = static_cast<T>(v.x) ;
    this->y = static_cast<T>(v.y) ;
  return *this ; }

  inline Vec2<T> operator+ (){ return Vec2<T>(*this) ; }
  inline Vec2<T> operator- (){ return Vec2<T>( - this->x, - this->y); }

  // *** OPERATOR [ANOTHER] ***
  // Vec before scalar, otherwise U = Vec2<W>

  inline Vec2<T> operator+ (const Vec2<T> &v){
    return Vec2<T>(this->x + v.x, this->y + v.y);
  }
  inline Vec2<T> operator+ (const T scalar){
    return Vec2<T>(this->x + scalar, this->y + scalar);
  }
  inline friend Vec2<T> operator+ (const T scalar, const Vec2<T> &v){
    return Vec2<T>(scalar + v.x, scalar + v.y);
  }
  inline friend Vec2<T> operator+ (const Vec2<T> &u, const Vec2<T> &v){
    return Vec2<T>(u.x + v.x, u.y + v.y);
  }


  inline Vec2<T> operator- (const Vec2<T> &v){
    return Vec2<T>(this->x - v.x, this->y - v.y);
  }
  inline Vec2<T> operator- (const T scalar){
    return Vec2<T>(this->x - scalar, this->y - scalar);
  }
  inline friend Vec2<T> operator- (const T scalar, const Vec2<T> &v){
    return Vec2<T>(scalar - v.x, scalar - v.y);
  }
  inline friend Vec2<T> operator- (const Vec2<T> &u, const Vec2<T> &v){
    return Vec2<T>(u.x - v.x, u.y - v.y);
  }

  inline Vec2<T> operator* (const Vec2<T> &v){
    return Vec2<T>(this->x * v.x, this->y * v.y);
  }
  inline Vec2<T> operator* (const T scalar){
    return Vec2<T>(this->x * scalar, this->y * scalar);
  }
  inline friend Vec2<T> operator* (const T scalar, const Vec2<T> &v){
    return Vec2<T>(scalar * v.x, scalar * v.y);
  }
  inline friend Vec2<T> operator* (const Vec2<T> &u, const Vec2<T> &v){
    return Vec2<T>(u.x * v.x, u.y * v.y);
  }

  inline Vec2<T> operator/ (const Vec2<T> &v){
    return Vec2<T>(this->x / v.x, this->y / v.y);
  }
  inline Vec2<T> operator/ (const T scalar){
    return Vec2<T>(this->x / scalar, this->y / scalar);
  }
  inline friend Vec2<T> operator/ (const T scalar, const Vec2<T> &v){
    return Vec2<T>(scalar / v.x, scalar / v.y);
  }
  inline friend Vec2<T> operator/ (const Vec2<T> &u, const Vec2<T> &v){
    return Vec2<T>(u.x / v.x, u.y / v.y);
  }

  inline Vec2<T> operator% (const T scalar){
    return Vec2<T>(this->x % scalar, this->y % scalar);
  }

  // *** BOOLEAN OPERATOR [ANOTHER] ***

  inline bool operator== (const Vec2<T> &v){
    return (this->x == v.x) && (this->y == v.y) ;
  }

  inline bool operator!= (const Vec2<T> &v){
    return (this->x != v.x) || (this->y != v.y) ;
  }

};


namespace func::vec2
{

  // *** ELEMENT-WISE MATH FUNCTION ***

  // abs, pow, sqrt

  template <typename T>
  inline BSIM::Vec2<T> abs(const BSIM::Vec2<T> &v){
    return BSIM::Vec2<T>( std::abs(v.x), std::abs(v.y) );
  }

  template <typename T, typename A>
  inline BSIM::Vec2<T> pow(const BSIM::Vec2<T> &v, const A scalar){
    return BSIM::Vec2<T>( std::pow(v.x, scalar), std::pow(v.y, scalar) );
  }

  template <typename T>
  inline BSIM::Vec2<T> sqrt(const BSIM::Vec2<T> &v){
    return BSIM::Vec2<T>( std::sqrt(v.x), std::sqrt(v.y) );
  }

  // ceil, floor, trunc, round

  template <typename T>
  inline BSIM::Vec2<T> ceil(const BSIM::Vec2<T> &v){
    return BSIM::Vec2<T>( std::ceil(v.x), std::ceil(v.y) );
  }

  template <typename T>
  inline BSIM::Vec2<T> floor(const BSIM::Vec2<T> &v){
    return BSIM::Vec2<T>( std::floor(v.x), std::floor(v.y) );
  }

  template <typename T>
  inline BSIM::Vec2<T> trunc(const BSIM::Vec2<T> &v){
    return BSIM::Vec2<T>( std::trunc(v.x), std::trunc(v.y) );
  }

  template <typename T>
  inline BSIM::Vec2<T> round(const BSIM::Vec2<T> &v){
    return BSIM::Vec2<T>( std::round(v.x), std::round(v.y) );
  }

  // isfinite, isinf, isnan, isnormal

  template <typename T>
  inline bool isfinite(const BSIM::Vec2<T> &v){
    return std::isfinite(v.x) && std::isfinite(v.y) ;
  }

  template <typename T>
  inline bool isinf(const BSIM::Vec2<T> &v){
    return std::isinf(v.x) || std::isinf(v.y) ;
  }

  template <typename T>
  inline bool isnan(const BSIM::Vec2<T> &v){
    return std::isnan(v.x) || std::isnan(v.y) ;
  }

  template <typename T>
  inline bool isnormal(const BSIM::Vec2<T> &v){
    return std::isnormal(v.x) && std::isnormal(v.y) ;
  }

  // max, min, pow

  template <typename T, typename A>
  inline BSIM::Vec2<T> max(const BSIM::Vec2<T> &v, const A scalar){
    const T scalarT = static_cast<T>(scalar) ;
    return BSIM::Vec2<T>( std::max(v.x, scalarT), std::max(v.y, scalarT) );
  }

  template <typename T>
  inline BSIM::Vec2<T> max(const BSIM::Vec2<T> &v1, const BSIM::Vec2<T> &v2){
    return BSIM::Vec2<T>( std::max(v1.x, v2.x), std::max(v1.y, v2.y) );
  }

  template <typename T, typename A>
  inline BSIM::Vec2<T> min(const BSIM::Vec2<T> &v, const A scalar){
    const T scalarT = static_cast<T>(scalar) ;
    return BSIM::Vec2<T>( std::min(v.x, scalarT), std::min(v.y, scalarT) );
  }

  template <typename T>
  inline BSIM::Vec2<T> min(const BSIM::Vec2<T> &v1, const BSIM::Vec2<T> &v2){
    return BSIM::Vec2<T>( std::min(v1.x, v2.x), std::min(v1.y, v2.y) );
  }

  template <typename T>
  inline BSIM::Vec2<T> pow(const BSIM::Vec2<T> &v, const BSIM::Vec2<T> &p){
    return BSIM::Vec2<T>( std::pow(v.x, p.x), std::pow(v.y, p.y) );
  }


  // exp, exp2, expm1, log, log10, log2, log1p
  // sin, cos, tan, asin, acos, atan, atan2
  // sinh, cosh, tanh, asinh, acosh, atanh
  // erf, erfc, tgamma, lgamma
  // ...

  template <typename T>
  inline BSIM::Vec2<T> apply(T (*func)(T), const BSIM::Vec2<T> &v){
    return BSIM::Vec2<T>( func(v.x), func(v.y) );
  }

  // length, normalize

  template <typename T>
  inline T length(const BSIM::Vec2<T> &v){
    return std::hypot(v.x, v.y);
  }

  template <typename T>
  inline BSIM::Vec2<T> normalize(const BSIM::Vec2<T> &v){
    return v/length(v) ;
  }

  // dot, cross, rotate_cc, rotate_c, distance

  template <typename T>
  inline T dot(const BSIM::Vec2<T> &v1, const BSIM::Vec2<T> &v2){
    return v1.x*v2.x + v1.y*v2.y ;
  }

  template <typename T>
  inline T cross(const BSIM::Vec2<T> &v1, const BSIM::Vec2<T> &v2){
    return v1.x*v2.y - v1.y*v2.x ;
  }

  template <typename T, typename A>
  BSIM::Vec2<T> rotate_cc(const BSIM::Vec2<T> &v, const A theta){
    T sin = std::sin( static_cast<T>(theta) );
    T cos = std::cos( static_cast<T>(theta) );
    return BSIM::Vec2<T>( v.x*cos - v.y*sin,
                    v.x*sin + v.y*cos );
  }

  template <typename T, typename A>
  inline BSIM::Vec2<T> rotate_c(const BSIM::Vec2<T> &v, const A theta){
    return rotate_cc(v, -theta);
  }

  template <typename T>
  inline T distance(const BSIM::Vec2<T> &v1, const BSIM::Vec2<T> &v2){
    return std::hypot(v1.x - v2.x, v1.y - v2.y);
  }


  // *** ADVANCED VECTOR FUNCTION ***

  // reflect, refract, faceforward, ...

} // NAMESPACE BSIM::func::vec2

} // NAMESPACE BSIM

#endif // BSIM_VEC2_HPP
