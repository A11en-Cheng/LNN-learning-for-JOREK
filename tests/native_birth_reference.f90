! Independent extraction of marker_spi_source.f90:809-817 real-kind semantics.
program birth_reference
  implicit none
  real*8 :: R_geo, P, P_norm, twopi
  integer :: n_plane
  real*4 :: dv, f
  read(*,*) R_geo, n_plane, P, P_norm
  twopi = 2.d0*acos(-1.d0)
  dv = 0.01 * 0.01 * R_geo * (twopi / n_plane)
  f = max(P*dv/P_norm, 0.e0)
  write(*,'(2es25.17)') real(dv,8), real(f,8)
end program birth_reference
