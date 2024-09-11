/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  root.cpp (main):
  Common parameters and common functions

\*****************************************************************************/

#include "root.hpp"

namespace BSIM
{

BSIM_root::BSIM_root()
{

}

BSIM_root::~BSIM_root()
{

}

BSIM::DateTime BSIM_root::starttime(2023, 1, 1, 0, 0, 0, true) ;
BSIM::DateTime BSIM_root::total_simulationtime(0, 0, 1, 0, 0, 0, false) ;
BSIM::DateTime BSIM_root::currenttime(2023, 1, 1, 0, 0, 0, true) ;
Uint           BSIM_root::timestep_second = 0 ;
Uint           BSIM_root::timestep_divide_second = 1 ;

// timestep_size (in seconds) = timestep_second/timestep_divide_second
// Iteration per second = timestep_divide_second

std::string    BSIM_root::input_dir      = "./BSIM_input/" ;
std::string    BSIM_root::metinfo_prefix = "metinfo_"      ;
std::string    BSIM_root::geoinfo_prefix = "geoinfo_"      ;
BSIM::DateTime BSIM_root::input_interval(0, 0, 0, 1, 0, 0, false) ;

std::string    BSIM_root::output_dir     = "./BSIM_output/" ;
std::string    BSIM_root::output_prefix  = "BSIM_out"       ;
BSIM::DateTime BSIM_root::output_interval(0, 0, 0, 1, 0, 0, false) ;

std::string    BSIM_root::checkpoint_prefix = "BSIM_save_" ;
BSIM::DateTime BSIM_root::checkpoint_interval(0, 0, 1, 0, 0, 0, false);

BSIM::Vec2<lpUint> BSIM_root::number_of_grid_point = BSIM::Vec2<lpUint>(1,1) ;
BSIM::Vec2<Float>  BSIM_root::grid_spacing         = BSIM::Vec2<Float>(1.0, 1.0) ;
BSIM::Vec2<Float>  BSIM_root::global_pos0          = BSIM::Vec2<Float>(0.0, 0.0) ;
BSIM::Vec2<lpUint> BSIM_root::shared_edge_size     = BSIM::Vec2<lpUint>(1,1) ;

lpUint   BSIM_root::boundary_type = 0 ; // 0 = external, 1 = periodic

BSIM::Vec2<lpUint> BSIM_root::num_process = BSIM::Vec2<lpUint>(1,1) ; // == NUM_MPI, MPU+I_SIZE
lpUint             BSIM_root::num_thread  = 1 ;                       // == OMP_NUM_THREADS
std::vector<Int>   BSIM_root::mpi_map ;

Uint        BSIM_root::iter_per_stdout = 1 ;
std::string BSIM_root::adios2file = "./BSIM_adios2_config.xml" ;

} // NAMESPACE BSIM
