/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  paralleldomain.hpp (header):
    Variables, memory allocation and domain decomposition

\*****************************************************************************/

#ifndef BSIM_PARALLELDOMAIN_HPP
#define BSIM_PARALLELDOMAIN_HPP

#include <mpi.h>
#include <omp.h>
#include <vector>

#include <root.hpp>
#include <io_adios2.hpp>
#include <cubicgrid2d.hpp>

// Use local ptr -->
// CubicGrid2D *xxx = BSIM_var[BSIM_VAR::tag::xxx] ;

namespace BSIM
{

class Parallel_Domain : private BSIM_root
{
public:
  Parallel_Domain(const Vec2<lpInt> domain_id);
  ~Parallel_Domain();

  const Vec2<lpInt> id ;   // Patch ID of the MPI process
  std::vector<CubicGrid2D*> BSIM_var ;

  // Remove if-else condition when deployed (single bc type)
  inline void exchange_all_bc(lpInt varid){
    if(this->boundary_type==0) exchange_all_bc_external(varid); else exchange_all_bc_periodic(varid) ;
  }
  inline void exchange_NS_bc(lpInt varid){
    if(this->boundary_type==0) exchange_NS_bc_external(varid); else exchange_NS_bc_periodic(varid) ;
  }
  inline void exchange_EW_bc(lpInt varid){
    if(this->boundary_type==0) exchange_EW_bc_external(varid); else exchange_EW_bc_periodic(varid) ;
  }
  inline void wait_all_bc(lpInt varid){
    if(this->boundary_type==0) wait_all_bc_external(varid); else wait_all_bc_periodic(varid) ;
  }
  inline void wait_NS_bc(lpInt varid){
    if(this->boundary_type==0) wait_NS_bc_external(varid); else wait_NS_bc_periodic(varid) ;
  }
  inline void wait_EW_bc(lpInt varid){
    if(this->boundary_type==0) wait_EW_bc_external(varid); else wait_EW_bc_periodic(varid) ;
  }

  IO_ADIOS2 *IO = nullptr ;
  bool adios2_read_checkpoint(const std::string filename);
  bool adios2_write_checkpoint(const std::string filename);
  bool adios2_write_output(const std::string filename);

private:
  lpInt num_req = 0 ;
  MPI_Request mpi_request[48] ; // (send,receive) x (sw,ss,se,ee,ne,nn,nw,ww) x (val, grad_x, grad_y)
  std::vector<Shared2D<Float>*> sendtobe_edge, receive_edge ;

  Vec2<lpInt> origin_on_global_grid ;  // Index on global grid that corresponding to local grid (0,0)
  //const int my_rank ;
  int rank_sw, rank_ss, rank_se, rank_ww, rank_ee, rank_nw, rank_nn, rank_ne ;
  int xx_size, xy_size, yy_size ;

  void exchange_all_bc_periodic(const lpInt varid);
  void exchange_NS_bc_periodic(const lpInt varid);
  void exchange_EW_bc_periodic(const lpInt varid);
  void wait_all_bc_periodic(const lpInt varid);
  void wait_NS_bc_periodic(const lpInt varid);
  void wait_EW_bc_periodic(const lpInt varid);
  void exchange_all_bc_external(const lpInt varid);
  void exchange_NS_bc_external(const lpInt varid);
  void exchange_EW_bc_external(const lpInt varid);
  void wait_all_bc_external(const lpInt varid);
  void wait_NS_bc_external(const lpInt varid);
  void wait_EW_bc_external(const lpInt varid);

protected:
  inline lpInt modulo(const lpInt a, const lpInt n){
    return static_cast<lpInt>( ((a % n) + n ) % n ) ;
  }
  lpInt rank_from_id_periodicBC(const lpInt idx, const lpInt idy);
  lpInt rank_from_id_externalBC(const lpInt idx, const lpInt idy);

  Vec2<lpInt> compute_range_local_domain(const lpInt proc_id, const lpInt num_proc, const lpInt stride, const lpInt nx);

  void allocate_variable(const Vec2<lpInt> grid_size, const Vec2<lpInt> edge_size, const Vec2<Float> pos0, const Vec2<Float> spacing);
  void deallocate_variable();

};

} // NAMESPACE BSIM

#endif // BSIM_PARALLELDOMAIN_HPP
