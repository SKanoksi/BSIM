/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  array2d.hpp:
    2D array datatype

\*****************************************************************************/

#ifndef BSIM_ARRAY2D_HPP
#define BSIM_ARRAY2D_HPP

//#include <ostream>
#include <iostream>

#include <shared2d.hpp>

// Note:
//  MPI   ==> Array2D.sendtobe --> Shared2D --> MPI_send | MPI_recv --> Array2D.copysharedfrom
// OpenMP ==> #pragma omp parallel for j --> j loop == j in Array2D.at(i,j) (y-axis)
// SIMD Vectorization ==> #pragma omp simd --> Nested loop i inside j loop where i in Array2D.at(i,j) (x-axis)

namespace BSIM
{

template <typename T> class Array2D
{
public:
  const Vec2<lpInt> size, edge, size_edge, full_size ;
  T *data = nullptr ;

  // *** CONSTRUCTOR ***
  // at, sendtobe_xx, copy_xx_from, copy_whole_from,
  // swap, fill, << (currently commented out)

  Array2D(const Vec2<lpInt> &core_size, const Vec2<lpInt> &edge_length)
         : size(core_size), edge(edge_length), size_edge(core_size+edge_length), full_size(core_size+2*edge_length)
  {
    this->data = new T[this->full_size.x * this->full_size.y] ;
    /*
    try{

      this->data = new T[this->full_size.x * this->full_size.y] ;

    }catch( const std::bad_alloc &e){

      std::cerr
      << "\nERROR while running BSIM::array2d.hpp \n"
      << "--> An array of data cannot be allocated, Out-of-memory?." << std::endl;

    }
    */
  }


  // Delete default copy and move operators
  Array2D(const Array2D&) = delete ;
  Array2D(Array2D&&) = delete ;
  Array2D& operator=(const Array2D&) = delete ;
  Array2D& operator=(Array2D&&) = delete;

  ~Array2D()
  {
    delete[] this->data ;
  }

  #pragma omp declare simd
  inline T &at(lpInt i, lpInt j){
    return this->data[this->full_size.x*(j+this->edge.y) + (i+this->edge.x)] ;
  }

  // w = west, e = east, n = north, s = south
  #pragma omp declare simd
  inline T &at_sw(lpInt i, lpInt j){
    return this->data[this->full_size.x*j + i] ;
  }
  #pragma omp declare simd
  inline T &at_ss(lpInt i, lpInt j){
    return this->data[this->full_size.x*j + (i+this->edge.x)] ;
  }
  #pragma omp declare simd
  inline T &at_se(lpInt i, lpInt j){
    return this->data[this->full_size.x*j + (i+this->size_edge.x)] ;
  }
  #pragma omp declare simd
  inline T &at_ee(lpInt i, lpInt j){
    return this->data[this->full_size.x*(j+this->edge.y) + (i+this->size_edge.x)] ;
  }
  #pragma omp declare simd
  inline T &at_ne(lpInt i, lpInt j){
    return this->data[this->full_size.x*(j+this->size_edge.y) + (i+this->size_edge.x)] ;
  }
  #pragma omp declare simd
  inline T &at_nn(lpInt i, lpInt j){
    return this->data[this->full_size.x*(j+this->size_edge.y) + (i+this->edge.x)] ;
  }
  #pragma omp declare simd
  inline T &at_nw(lpInt i, lpInt j){
    return this->data[this->full_size.x*(j+this->size_edge.y) + i] ;
  }
  #pragma omp declare simd
  inline T &at_ww(lpInt i, lpInt j){
    return this->data[this->full_size.x*(j+this->edge.y) + i] ;
  }

  // *** sendtobe_xx ***
  void sendtobe_edge(Shared2D<T> &u)
  {
    this->sendtobe_part(u.ne, this->edge.x, this->edge.y, this->edge.x, this->edge.y);
    this->sendtobe_part(u.nn, this->edge.x, this->edge.y, this->size.x, this->edge.y);
    this->sendtobe_part(u.nw, this->size.x, this->edge.y, this->edge.x, this->edge.y);

    this->sendtobe_part(u.ee, this->edge.x, this->edge.y, this->edge.x, this->size.y);
    this->sendtobe_part(u.ww, this->size.x, this->edge.y, this->edge.x, this->size.y);

    this->sendtobe_part(u.se, this->edge.x, this->size.y, this->edge.x, this->edge.y);
		this->sendtobe_part(u.ss, this->edge.x, this->size.y, this->size.x, this->edge.y);
    this->sendtobe_part(u.sw, this->size.x, this->size.y, this->edge.x, this->edge.y);
  }


  inline void sendtobe_sw(Shared2D<T> &u){
    this->sendtobe_part(u.sw, this->size.x, this->size.y, this->edge.x, this->edge.y);
  }
  inline void sendtobe_ss(Shared2D<T> &u){
    this->sendtobe_part(u.ss, this->edge.x, this->size.y, this->size.x, this->edge.y);
  }
  inline void sendtobe_se(Shared2D<T> &u){
    this->sendtobe_part(u.se, this->edge.x, this->size.y, this->edge.x, this->edge.y);
  }
  inline void sendtobe_ww(Shared2D<T> &u){
    this->sendtobe_part(u.ww, this->size.x, this->edge.y, this->edge.x, this->size.y);
  }
  inline void sendtobe_ee(Shared2D<T> &u){
    this->sendtobe_part(u.ee, this->edge.x, this->edge.y, this->edge.x, this->size.y);
  }
  inline void sendtobe_nw(Shared2D<T> &u){
    this->sendtobe_part(u.nw, this->size.x, this->edge.y, this->edge.x, this->edge.y);
  }
  inline void sendtobe_nn(Shared2D<T> &u){
    this->sendtobe_part(u.nn, this->edge.x, this->edge.y, this->size.x, this->edge.y);
  }
  inline void sendtobe_ne(Shared2D<T> &u){
    this->sendtobe_part(u.ne, this->edge.x, this->edge.y, this->edge.x, this->edge.y);
  }


  void sendtobe_part(T *ptrOut, lpInt sX, lpInt sY, lpInt nX, lpInt nY)
  {
    for(lpInt j=0 ; j != nY ; ++j)
    {
      // Loop expected to be vectorized
      #pragma omp simd simdlen(SIMD_NUM_INST)
      for(lpInt i=0 ; i != (nX - nX%SIMD_NUM_INST) ; ++i)
      {
        ptrOut[nX*j + i] = this->data[this->full_size.x*(j+sY) + (i+sX)] ;
      }
      // Remaining
      if( nX%SIMD_NUM_INST ){
        for(lpInt i=(nX - nX%SIMD_NUM_INST) ; i != nX ; ++i)
        {
          ptrOut[nX*j + i] = this->data[this->full_size.x*(j+sY) + (i+sX)] ;
        }
      }
    }
  }


  // *** copy_xx_from ***
  void copy_edge_from(Shared2D<T> &u)
  {
    this->copysharedfrom_part(u.sw, 0,                                 0, this->edge.x, this->edge.y);
    this->copysharedfrom_part(u.ss, this->edge.x,                      0, this->size.x, this->edge.y);
    this->copysharedfrom_part(u.se, this->size_edge.x,                 0, this->edge.x, this->edge.y);

    this->copysharedfrom_part(u.ww, 0,                      this->edge.y, this->edge.x, this->size.y);
    this->copysharedfrom_part(u.ee, this->size_edge.x,      this->edge.y, this->edge.x, this->size.y);

    this->copysharedfrom_part(u.nw, 0,                 this->size_edge.y, this->edge.x, this->edge.y);
    this->copysharedfrom_part(u.nn, this->edge.x,      this->size_edge.y, this->size.x, this->edge.y);
    this->copysharedfrom_part(u.ne, this->size_edge.x, this->size_edge.y, this->edge.x, this->edge.y);
  }


  inline void copy_sw_from(Shared2D<T> &u){
    this->copysharedfrom_part(u.sw, 0, 0, this->edge.x, this->edge.y);
  }
  inline void copy_ss_from(Shared2D<T> &u){
    this->copysharedfrom_part(u.ss, this->edge.x, 0, this->size.x, this->edge.y);
  }
  inline void copy_se_from(Shared2D<T> &u){
    this->copysharedfrom_part(u.se, this->size_edge.x, 0, this->edge.x, this->edge.y);
  }
  inline void copy_ww_from(Shared2D<T> &u){
    this->copysharedfrom_part(u.ww, 0, this->edge.y, this->edge.x, this->size.y);
  }
  inline void copy_ee_from(Shared2D<T> &u){
    this->copysharedfrom_part(u.ee, this->size_edge.x, this->edge.y, this->edge.x, this->size.y);
  }
  inline void copy_nw_from(Shared2D<T> &u){
    this->copysharedfrom_part(u.nw, 0, this->size_edge.y, this->edge.x, this->edge.y);
  }
  inline void copy_nn_from(Shared2D<T> &u){
    this->copysharedfrom_part(u.nn, this->edge.x, this->size_edge.y, this->size.x, this->edge.y);
  }
  inline void copy_ne_from(Shared2D<T> &u){
    this->copysharedfrom_part(u.ne, this->size_edge.x, this->size_edge.y, this->edge.x, this->edge.y);
  }


  void copysharedfrom_part(T *ptrOut, lpInt sX, lpInt sY, lpInt nX, lpInt nY)
  {
    for(lpInt j=0 ; j != nY ; ++j)
    {
      // Loop expected to be vectorized
      #pragma omp simd simdlen(SIMD_NUM_INST)
      for(lpInt i=0 ; i != (nX - nX%SIMD_NUM_INST) ; ++i)
      {
        this->data[this->full_size.x*(j+sY) + (i+sX)] = ptrOut[nX*j + i] ;
      }
      // Remaining
      if( nX%SIMD_NUM_INST ){
        for(lpInt i=(nX - nX%SIMD_NUM_INST) ; i != nX ; ++i)
        {
          this->data[this->full_size.x*(j+sY) + (i+sX)] = ptrOut[nX*j + i] ;
        }
      }
    }
  }


  void copy_whole_from(Array2D<T> &u)
  {
    hpInt num = this->full_size.x*this->full_size.y ;

    // Loop expected to be vectorized
    #pragma omp simd simdlen(SIMD_NUM_INST)
    for(hpInt n=0 ; n != (num - num%SIMD_NUM_INST) ; ++n)
    {
      this->data[n] = u.data[n] ;
    }
    // Remaining
    if( num%SIMD_NUM_INST ){
      for(hpInt n=(num - num%SIMD_NUM_INST) ; n != num ; ++n)
      {
        this->data[n] = u.data[n] ;
      }
    }
  }

  /*
  static void swap(Array2D<T> &u, Array2D<T> &v)
  {
    T *temp ;
    temp = v.data ;
    v.data = u.data ;
    u.data = temp ;
  }


  void fill(T value)
  {
    hpInt num = this->full_size.x * this->full_size.y ;

    // Loop expected to be vectorized
    #pragma omp simd simdlen(SIMD_NUM_INST)
    for(lpInt n=0 ; n != (num - num%SIMD_NUM_INST) ; ++n)
    {
      this->data[n] = value ;
    }
    // Remaining
    if( num%SIMD_NUM_INST ){
      for(lpInt n=(num - num%SIMD_NUM_INST) ; n != num ; ++n)
      {
        this->data[n] = value ;
      }
    }
  }

  inline void fill_west(T value){
    this->fill_part(value, 0, 0, this->edge.x, this->full_size.y);
  }

  inline void fill_south(T value){
    this->fill_part(value, 0, 0, this->full_size.x, this->edge.y);
  }

  inline void fill_east(T value){
    this->fill_part(value, this->size_edge.x, 0, this->edge.x, this->full_size.y);
  }

  inline void fill_north(T value){
    this->fill_part(value, 0, this->size_edge.y, this->full_size.x, this->edge.y);
  }

  inline void fill_core(T value){
    this->fill_part(value, this->edge.x, this->edge.y, this->size.x, this->size.y);
  }

  void fill_part(T value, lpInt startX, lpInt startY, lpInt nX, lpInt nY)
  {
    for(lpInt iy=startY ; iy != startY+nY ; ++iy)
    {
      // Loop expected to be vectorized
      #pragma omp simd simdlen(SIMD_NUM_INST)
      for(lpInt ix=0 ; ix != (nX - nX%SIMD_NUM_INST) ; ++ix)
      {
        this->data[this->full_size.x*iy + ix + startX] = value ;
      }
      // Remaining
      if( nX%SIMD_NUM_INST ){
        for(lpInt ix=(nX - nX%SIMD_NUM_INST) ; ix != nX ; ++ix)
        {
          this->data[this->full_size.x*iy + ix + startX] = value ;
        }
      }
    }
  }
  */

  // *** OSTREAM ***
  friend std::ostream &operator<< (std::ostream &os, Array2D<T> &u)
  {
    for(lpInt j = u.size.y+u.edge.y-1 ; j != -u.edge.y-1 ; --j)
    {
      for(lpInt i = -u.edge.x ; i != u.size.x+u.edge.x ; ++i)
      {
        os << u.at(i,j) << " " ;
      }
      os << "\n" ;
    }

  return os; }

};

} // NAMESPACE BSIM

#endif // BSIM_ARRAY2D_HPP
