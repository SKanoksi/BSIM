/*****************************************************************************\

  2D Burgers' simulation

  Version 0.1.0
  Copyright (c) 2024, Somrath Kanoksirirath <somrathk@gmail.com>
  All rights reserved under BSD 3-clause license.
*******************************************************************************

  dynamics.hpp (header):
    Dynamics procedures of BSIM main loop

\*****************************************************************************/

#ifndef DYNAMICS_HPP
#define DYNAMICS_HPP

#include <cubicgrid2d.hpp>

// To be check:
// 1. source-sink    == source
// 2. growth-decay   == growth
// 3. advection      == advect
// 4. diffusion      == diffuse

namespace BSIM::CONST
{

constexpr hpFloat PI =3.14159265358979323846264338327950 ;

} // NAMESPACE SBIM::CONST


namespace BSIM_DYNAMICS
{



// Source-Sink
// 1. Add (+) s*dt to val (only val)

// Growth-Decay
// 1. Multiply (*) std::exp(g*dt) to val

// Source + Growth-Decay
// 1. Multiply (*) std::exp(g*dt) this factor to val
// 2. Add (+) (s/g)*( std::exp(g*dt) - 1 ) to val


// Advection -- Semi-Lagrangian view
// 1. Calculate x_depart
//   dx*ix - u*dt ;
//   dy*jy - v*dt ;
// 2. Apply BC
// 3. Forward with conservation condition
// Note: local_x_depart = local_x_unequal[j]
//       dx_unequal = (local_x_unequal[i_depart+1] - local_x_unequal[i_depart-1])/2
//       u_grad = either at i_arrival (on grid point) or at i_depart (require additional interpolation)
//                --> use u_grad at i_arrival
BSIM::Vec2<Float> forwardX_advect(BSIM::CubicGrid2D *var,
                                  const Float local_x_depart, const lpInt jy,
                                  const Float dt, const Float dx_unequal, const Float u_grad, Float *temp);
BSIM::Vec2<Float> forwardY_advect(BSIM::CubicGrid2D *var,
                                  const lpInt ix, const Float local_y_depart,
                                  const Float dt, const Float dy_unequal, const Float v_grad, Float *temp);

// Diffusion
// 1. Calculate distance = (i_from - i_to)*grid_spacing = from (ix,jy) to (i,j)
//    Ddtx4 = 4*D*dt
// 2. Calculate the total contribution from several (ix,jy) to (i,j), using forward_diffusion
BSIM::Vec2<Float> forward_diffuse(const hpFloat Ddtx4, const Float spacing, const Float distance, Float *coeff);

// Burgers'
void diffuse_in_hopfcole_cancelcommon(const hpFloat Kdtx4, const Float spacing, const Float distance, Float *temp, hpFloat *workspace);

// -------------------------------------------

namespace func
{

inline hpFloat inverse_hopfcole_val(const hpFloat P, const hpFloat Px, const hpFloat Kdt){
  return -Px/std::abs(P)/Kdt ;
}

inline hpFloat inverse_hopfcole_grad(const hpFloat P, const hpFloat Px, const hpFloat Pxx, const hpFloat Kdt){
  return 0.5*(- Pxx + Px*Px/P)/std::abs(P)/Kdt/Kdt ;
}

hpFloat integrate_PolydegNGaussian_c4(const hpFloat x0, const hpFloat l, const hpFloat b, const Float *cN);
hpFloat integrate_PolydegNGaussian_c5(const hpFloat x0, const hpFloat l, const hpFloat b, const Float *cN);

hpFloat integrate_PolydegNGaussian_c8_mid(const hpFloat x0, const hpFloat l, const hpFloat b, const hpFloat *cN);
hpFloat integrate_PolydegNGaussian_c9_mid(const hpFloat x0, const hpFloat l, const hpFloat b, const hpFloat *cN);
hpFloat integrate_PolydegNGaussian_c10_mid(const hpFloat x0, const hpFloat l, const hpFloat b, const hpFloat *cN);

} // NAMESPACE FUNC

} // NAMESPACE BSIM_DYNAMICS

#endif // DYNAMICS_HPP
