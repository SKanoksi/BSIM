/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  datatype.hpp (header):
    standard internal datatypes

\*****************************************************************************/

#ifndef BSIM_DATATYPE_HPP
#define BSIM_DATATYPE_HPP

#include <cstdint>
#include <string>

// Optimization --> GNU with -fopt-info-vec-missed
#define SIMD_NUM_INST 4

// Future: When the fixed-point float feature in C++23 is widely supported (GCC 13++).
// --> #include <stdfloat> --> std::float128_t, ...
using Float   = double ;  // main datatype
using lpFloat = float ;
using hpFloat = long double ;
#define MPI_Float   MPI_DOUBLE
#define MPI_lpFloat MPI_FLOAT
#define MPI_hpFloat MPI_LONG_DOUBLE

using Int   = std::int32_t ;
using lpInt = std::int16_t ; // change to std::int32_t to avoid int-converstion
using hpInt = std::int64_t ;
#define MPI_Int   = MPI_INT32_T
#define MPI_lpInt = MPI_INT32_T
#define MPI_hpInt = MPI_INT64_T

using Uint   = std::uint32_t ;
using lpUint = std::uint16_t ;
using hpUint = std::uint64_t ;
#define MPI_Uint   MPI_UINT32_T
#define MPI_lpUint MPI_UINT16_T
#define MPI_hpUint MPI_UINT64_T

template <typename U>
inline std::string num2str(const U num){ return std::to_string(num); }

Float str2Float(const std::string str);
lpFloat str2lpFloat(const std::string str);
hpFloat str2hpFloat(const std::string str);

Int str2Int(const std::string str);
lpInt str2lpInt(const std::string str);
hpInt str2hpInt(const std::string str);

Uint str2Uint(const std::string str);
lpUint str2lpUint(const std::string str);
hpUint str2hpUint(const std::string str);

// https://en.cppreference.com/w/cpp/string/basic_string/stol

#endif // BSIM_DATATYPE_HPP
