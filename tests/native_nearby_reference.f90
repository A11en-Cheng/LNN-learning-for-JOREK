! Test-only data and full tensor-cubic geometry. The locator and neighbour
! transformations are compiled separately from the actual JOREK sources.
module data_structure
  implicit none
  integer, parameter :: n_vertex_max=4
  type type_node_list
    integer :: unused=0
  end type
  type type_element
    integer :: vertex(4), neighbours(4)
    real*8 :: polynomial(2,0:3,0:3)
  end type
  type type_element_list
    integer :: n_elements
    type(type_element), allocatable :: element(:)
  end type
end module

module oracle_observations
  implicit none
  integer :: fallback_calls=0
  integer :: fallback_element=0, fallback_flag=99
  real*8 :: fallback_st(2)=[0.25d0,0.75d0]
end module

module mod_interp
  use data_structure
  implicit none
contains
  pure subroutine interp_RZP(nodes,elements,e,s,t,phi,R,Rs,Rt,Rp,Z,Zs,Zt,Zp)
    type(type_node_list), intent(in) :: nodes
    type(type_element_list), intent(in) :: elements
    integer, intent(in) :: e
    real*8, intent(in) :: s,t,phi
    real*8, intent(out) :: R,Rs,Rt,Rp,Z,Zs,Zt,Zp
    real*8 :: x(2),xs(2),xt(2),c(2)
    integer :: i,j
    x=0; xs=0; xt=0
    do j=0,3
      do i=0,3
        c=elements%element(e)%polynomial(:,i,j)
        x=x+c*s**i*t**j
        if(i>0) xs=xs+c*i*s**(i-1)*t**j
        if(j>0) xt=xt+c*j*s**i*t**(j-1)
      enddo
    enddo
    R=x(1); Z=x(2); Rs=xs(1); Zs=xs(2); Rt=xt(1); Zt=xt(2)
    Rp=0; Zp=0
  end subroutine
end module

! Deliberately controlled fallback seam, NOT a replacement find_RZ oracle.
subroutine find_RZ(nodes,elements,R,Z,Rout,Zout,e,s,t,flag)
  use data_structure
  use oracle_observations
  implicit none
  type(type_node_list), intent(in) :: nodes
  type(type_element_list), intent(in) :: elements
  real*8, intent(in) :: R,Z
  real*8, intent(out) :: Rout,Zout,s,t
  integer, intent(inout) :: e
  integer, intent(out) :: flag
  fallback_calls=fallback_calls+1
  e=fallback_element; flag=fallback_flag
  s=fallback_st(1); t=fallback_st(2); Rout=R; Zout=Z
end subroutine

program native_nearby_reference
  use data_structure
  use mod_find_rz_nearby
  use mod_neighbours
  use mod_interp
  use oracle_observations
  implicit none
  type(type_node_list) :: nodes
  type(type_element_list) :: elements
  integer :: n,e,i,j,mode,old,flag,out
  real*8 :: st(2),target(2),oldx(2),s,t,dummy(6)
  read(*,*) mode,n
  elements%n_elements=n
  allocate(elements%element(n))
  do e=1,n
    read(*,*) elements%element(e)%vertex
    read(*,*) elements%element(e)%neighbours
    do j=0,3
      do i=0,3
        read(*,*) elements%element(e)%polynomial(:,i,j)
      enddo
    enddo
  enddo
  read(*,*) old,st,target
  read(*,*) fallback_element,fallback_flag,fallback_st
  if(mode==1) then
    call coord_in_neighbour(nodes,elements,old,out,st)
    flag=0
    s=st(1); t=st(2)
  else
    oldx=0
    if(old>=1.and.old<=n) then
      call interp_RZP(nodes,elements,old,st(1),st(2),0d0, &
        oldx(1),dummy(1),dummy(2),dummy(3),oldx(2),dummy(4),dummy(5),dummy(6))
    endif
    call find_RZ_nearby(nodes,elements,oldx(1),oldx(2),st(1),st(2),old, &
      target(1),target(2),s,t,out,flag)
  endif
  write(*,'(3i8,4es26.17)') out,flag,fallback_calls,s,t,target
end program
