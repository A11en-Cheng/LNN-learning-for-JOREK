! Small independent oracle, not a JOREK solver/build. Read-only source excerpts:
! mod_integrals3D.f90:438-441,452-454,671-686,1103-1108,1131-1146,
! 1740-1765,2531-2557; equil_info.f90:982-1024.
! Input: independently evaluated cubic Gauss psi,Z,rho,rhoimp,Te,P4,P5,volume.
! Explicit diagnostic candidate: rank0 current ES, other ranks prior ES;
! use_marker AND present(aux_node_list). No density/Te floor in shell sums.
program nonlocal_reference
  implicit none
  integer :: nf,nt,nq,ncpu,per_element,chunk,iq,is,i,j,k
  real*8 :: axis(2),bnd(2),xp(2,2),zp(2,2),ratio,f,ne,density,charge,l1,l2
  real*8 :: row(8),vol_weight,query
  real*8, allocatable :: coarse(:),volume(:),number(:),energy(:)
  read(*,*) nf,nt,nq,ncpu,per_element
  read(*,*) axis(1),bnd(1),xp(:,1),zp(:,1)
  read(*,*) axis(2),bnd(2),xp(:,2),zp(:,2)
  read(*,*) ratio
  allocate(coarse(nf),volume(nf),number(nf),energy(nf))
  volume=0.d0;number=0.d0;energy=0.d0
  chunk=ceiling(float(nq/per_element)/ncpu)
  do iq=1,nq
    read(*,*) row
    k=1
    if ((iq-1)/per_element>=chunk) k=2
    do is=1,nf
      coarse(is)=axis(k)+float(is-1)/nf*(bnd(k)-axis(k))
    end do
    f=flux(row(1),row(2),axis(k),bnd(k),xp(:,k),zp(:,k))
    if (f<0.d0.or.f>=1.d0) cycle
    ! These are two additive source branches, not an invented charge factor.
    charge=row(6)
    density=row(7)
    charge=charge+row(6)
    density=density+row(7)
    charge=max(charge,0.d0)
    l1=1.d-3*1.d-1;l2=1.d-3*1.d-1
    if (density<l1+l2) density=l1+l2*exp((density-l1-l2)/l2)
    charge=min(charge/density,10.d0)
    ne=row(3)+(ratio*charge-1.d0)*row(4)
    if (f>=plain(coarse(nf),axis(k),bnd(k))) then
      is=nf
    else
      is=int(f*real(nf,8))+1
      if (f<plain(coarse(is),axis(k),bnd(k))) then
        do while (f<plain(coarse(is),axis(k),bnd(k)).and.is>1)
          is=is-1
        end do
      else
        do while (f>=plain(coarse(is+1),axis(k),bnd(k)).and.is<nf-1)
          is=is+1
        end do
      end if
    end if
    vol_weight=row(8)
    volume(is)=volume(is)+vol_weight
    number(is)=number(is)+ne*vol_weight
    energy(is)=energy(is)+ne*row(5)*vol_weight
  end do
  if (any(volume<=0.d0).or.any(number<=0.d0)) stop 2
  energy=energy/number
  number=number/volume
  do is=1,nf
    coarse(is)=axis(1)+float(is-1)/nf*(bnd(1)-axis(1))
  end do
  do is=1,nt
    query=axis(1)+float(is-1)/nt*(bnd(1)-axis(1))
    i=minloc(abs(coarse-query),dim=1)
    if (abs(query-axis(1))>=abs(coarse(i)-axis(1))) then
      j=i+1
    else
      j=i-1
    end if
    if (j>nf) j=nf-1
    write(*,'(3es26.17)') query, &
      (energy(i)-energy(j))/(coarse(i)-coarse(j))*(query-coarse(i))+energy(i), &
      (number(i)-number(j))/(coarse(i)-coarse(j))*(query-coarse(i))+number(i)
  end do
contains
  pure real*8 function plain(psi,a,b)
    real*8,intent(in)::psi,a,b
    plain=(psi-a)/(b-a)
  end function
  pure real*8 function flux(psi,z,a,b,x,zx)
    real*8,intent(in)::psi,z,a,b,x(2),zx(2)
    real*8::lower,upper
    flux=plain(psi,a,b)
    lower=plain(x(1),a,b);upper=plain(x(2),a,b)
    if (flux<lower.and.z<zx(1)) flux=2.d0*lower-flux
    if (flux<upper.and.z>zx(2)) flux=2.d0*upper-flux
  end function
end program
