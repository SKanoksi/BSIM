/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  main.cpp:
    Top-most level, Shortest workflow

\*****************************************************************************/

#include <omp.h>
#include <mpi.h>

#include <iostream>
#include <string>
#include <filesystem>
#include <chrono>

#include <parser.hpp>
#include <simulation.hpp>


int mpi_rank, mpi_size ;

inline bool is_file_exist(std::string path){
  return std::filesystem::exists(std::filesystem::path(path));
}


bool run_BSIM_model(BSIM::Vec2<lpInt> id)
{
  BSIM_model *model = new BSIM_model(id);

  {
    std::string filename = model->input_dir
                           + model->checkpoint_prefix
                           + model->starttime.to_str() ;

    if( mpi_rank == mpi_size-1 ){
      std::cout << "BSIM reads first state :: from " << filename << std::endl;
    }

    if( !is_file_exist(filename) ){
      std::cerr
      << "\nERROR while running BSIM::main.cpp from MPI rank " << mpi_rank << " " << model->domain->id << "\n"
      << "--> initial checkpoint file \"" << filename << "\" was not found."
      << std::endl;
      delete model ;
      return false ;
    }
    if( !model->domain->adios2_read_checkpoint(filename) ){
      delete model ;
      return false ;
    }

    // Apply BC = initial
    for(lpInt varid=0 ; varid != static_cast<lpInt>(model->domain->BSIM_var.size()) ; ++varid)
    {
      model->domain->exchange_all_bc(varid);
      model->domain->wait_all_bc(varid);
    }

  }

  const Uint nt = static_cast<Uint>( std::ceil( static_cast<Float>(model->total_simulationtime.to_second())/model->dt ) );
  const Uint iter_per_checkpoint = static_cast<Uint>( std::ceil( static_cast<Float>(model->checkpoint_interval.to_second())/model->dt ) );
  const Uint iter_per_output     = static_cast<Uint>( std::ceil( static_cast<Float>(model->output_interval.to_second())/model->dt ) );
  const std::string checkpoint_file_prefix = model->output_dir + model->checkpoint_prefix ;
  const std::string     output_file_prefix = model->output_dir + model->output_prefix     ;

  bool normal_exit = true ;
  #pragma omp parallel default(shared)
  {
  if( model->num_thread != omp_get_num_threads() )
  {
    std::cerr
    << "\nERROR while running BSIM::main.cpp from MPI rank " << mpi_rank << " " << model->domain->id << "\n"
    << "--> Detected number of OpenMP threads, i.e., OMP_NUM_THREADS, (" << omp_get_num_threads()
    << ") != num_thread (" << model->num_thread
    << ") specified in runtime.setup.BSIM file" << std::endl ;
    normal_exit = false ;
  }
  #pragma omp master
  {
    for(Uint i = 1 ; i != nt+1 ; ++i)
    {
      if( !normal_exit ){ break; }

      // Simulation
      normal_exit = model->forward_all();

      // Forward model time
      if( i%model->timestep_divide_second == 0 ){
        model->currenttime += model->timestep_second ;
      }

      // Log
      if( i%model->iter_per_stdout==0 && mpi_rank==mpi_size-1 ){
        std::cout
        << "BSIM simulation time   :: "
        << model->currenttime.to_str()
        << " :: Iteration " << i << "/" << nt << std::endl;
      }

      // Output
      if( i%iter_per_output==0 && i!=0 ){
        std::string filename = output_file_prefix + model->currenttime.to_str() ;

        if( mpi_rank == mpi_size-1 ){
          std::cout
          << "BSIM writes output     :: to " << filename << std::endl;
        }

        if( !model->domain->adios2_write_output(filename) ){ normal_exit = false ; }
      }

      // Checkpoint
      if( i%iter_per_checkpoint==0 && i!=0 ){
        std::string filename = checkpoint_file_prefix + model->currenttime.to_str() ;

        if( mpi_rank == mpi_size-1 ){
          std::cout
          << "BSIM writes checkpoint :: to " << filename << std::endl;
        }

        if( !model->domain->adios2_write_checkpoint(filename) ){ normal_exit = false ; }
      }

    } // TIME ITERATION
  } // OMP MASTER
  } // OMP PARALLEL

  delete model ;

return (normal_exit) ? true : false ; }


void display_BSIM_usage(){
  std::cout
  << "Usage: BSIM [OPTIONS] \n"
  << "\n"
  << "A numerical weather prediciton model \n"
  << "  Version 0.0.4 -- 01 Oct 2023\n"
  << "\n"
  <<"OPTIONS: \n"
  <<"  -i,--setup \n"
  <<"      path to runtime.setup.BSIM file (default: ./runtime.setup.BSIM) \n"
  <<"  -h,--help \n"
  <<"      show this help message and exit \n"
  << std::endl ;
}


int main(int argc, char* argv[]){

  bool run_BSIM = true ;

  int mpi_thread_support ;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_thread_support);
  if( mpi_thread_support < MPI_THREAD_MULTIPLE )
  {
    std::cout
    << "\nWARNING while running BSIM::main.cpp \n"
    << "--> MPI_THREAD_MULTIPLE (level 4) was requested."
    << "--> The current level is " << mpi_thread_support << "." << std::endl;
  }

  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  // 0. Parsing input argument
  std::string setupfile = "./runtime.setup.BSIM" ;

  for(lpInt i=1 ; i != static_cast<lpInt>(argc) ; ++i){
    std::string argin = argv[i] ;
    if( argin == "-h" || argin == "--help" ){
      if( mpi_rank == mpi_size-1 )
      {
        display_BSIM_usage();
      }
      run_BSIM = false ;
      break;
    }else if( argin == "-i" || argin == "--setup" ){
      ++i ;
      setupfile = argv[i] ;
      continue;
    }else{
      if( mpi_rank == mpi_size-1 )
      {
        std::cerr
        << "\nERROR while running BSIM::main.cpp \n"
        << "--> Unknown option :: " << argin << std::endl;
      }
      run_BSIM = false ;
      break;
    }
  }

  bool global_run_BSIM ;
  MPI_Allreduce(&run_BSIM, &global_run_BSIM, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);

  // 1. Parsing setup file
  if( global_run_BSIM )
  {
    BSIM::Vec2<lpInt> domainID ;
    BSIM::Parser parser ;
    if( is_file_exist(setupfile) ){
      if( !parser.parse_setup(setupfile.c_str()) ){
        std::cerr
        << "\nERROR while running BSIM::main.cpp from MPI rank " << mpi_rank << "\n"
        << "--> Cannot parse the input setup file, \"" << setupfile << "\"."
        << std::endl;
        run_BSIM = false ;
      }
      if( !is_file_exist(parser.adios2file) ){
        std::cerr
        << "\nERROR while running BSIM::main.cpp from MPI rank " << mpi_rank << "\n"
        << "--> The ADIOS2 config file, \"" << parser.adios2file << "\", was not found."
        << std::endl;
        run_BSIM = false ;
      }
      if( static_cast<lpUint>(mpi_size) != parser.num_process.x * parser.num_process.y )
      {
        std::cerr
        << "\nERROR while running BSIM::main.cpp \n"
        << "--> Allocated number of MPI processes (" << mpi_size
        << ") != num_process.x * num_process.y (" << parser.num_process.x * parser.num_process.y
        << ") specified in runtime.setup.BSIM file" << std::endl ;
        run_BSIM = false ;
      }else if( !parser.get_domain_id(mpi_rank, domainID) ) // Important
      {
        std::cerr
        << "\nERROR while running BSIM::main.cpp \n"
        << "--> Domain ID of MPI rank " << mpi_rank << " was not found in \"mpi_map\". "
        << "Recheck the runtime.setup.BSIM file" << std::endl ;
        run_BSIM = false ;
      }
    }else{
      std::cerr
      << "\nERROR while running BSIM::main.cpp from MPI rank " << mpi_rank << "\n"
      << "--> The input setup file, \"" << setupfile << "\", was not found."
      << std::endl;
      run_BSIM = false ;
    }

    MPI_Allreduce(&run_BSIM, &global_run_BSIM, 1, MPI_CXX_BOOL, MPI_LAND, MPI_COMM_WORLD);

    // 2. Run main program

    if( global_run_BSIM )
    {
      if( mpi_rank == mpi_size-1 )
      {
        parser.print_setting();
        std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now() ;
        std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
        std::cout
        << " -- Start running BSIM model -- \n"
        << "    " << std::ctime(&timestamp) << std::endl;
        std::cout.flush();
      }

      if( !run_BSIM_model(domainID) )
      {
        std::cerr
        << "\nERROR while running BSIM::main.cpp from MPI rank " << mpi_rank << "\n"
        << "--> Abnormal exit from BSIM main loop."
        << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);

      }else if( mpi_rank == mpi_size-1 ){
        std::chrono::time_point<std::chrono::system_clock> now = std::chrono::system_clock::now() ;
        std::time_t timestamp = std::chrono::system_clock::to_time_t(now);
        std::cout
        << "\n    " << std::ctime(&timestamp)
        << " -- Finish running BSIM model -- " << std::endl;
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

return 0; }
