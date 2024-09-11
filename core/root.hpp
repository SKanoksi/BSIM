/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  root.hpp (header):
    Common parameters and common functions

\*****************************************************************************/

#ifndef BSIM_ROOT_HPP
#define BSIM_ROOT_HPP

#include <datatype.hpp>
#include <vec2.hpp>
#include <datetime.hpp>
#include <vector>

namespace BSIM
{

class BSIM_root
{
public:
  BSIM_root();
  ~BSIM_root();

  static BSIM::DateTime starttime ;
  static BSIM::DateTime total_simulationtime ;
  static BSIM::DateTime currenttime ;
  static Uint           timestep_second ;
  static Uint           timestep_divide_second ;

  static std::string    input_dir      ;
  static std::string    metinfo_prefix ;
  static std::string    geoinfo_prefix ;
  static BSIM::DateTime input_interval ;

  static std::string    output_dir      ;
  static std::string    output_prefix   ;
  static BSIM::DateTime output_interval ;

  static std::string    checkpoint_prefix   ;
  static BSIM::DateTime checkpoint_interval ;

  static BSIM::Vec2<lpUint> number_of_grid_point ;
  static BSIM::Vec2<Float>  grid_spacing         ;
  static BSIM::Vec2<Float>  global_pos0          ;
  static BSIM::Vec2<lpUint> shared_edge_size     ;

  static lpUint boundary_type ;

  static BSIM::Vec2<lpUint> num_process ;
  static lpUint             num_thread  ;
  static std::vector<Int>   mpi_map ;     // Temporary

  static Uint iter_per_stdout ;
  static std::string adios2file ;

};

}

#endif // BSIM_ROOT_HPP
