! Independent component driver linked to the unmodified JOREK basis + Gauss modules.
! Axisymmetric model502 fixture only; not the nine-field matrix or a fluid solve.
program native_reference
  use mod_basisfunctions
  use gauss
  implicit none
  real*8 :: sz(4,4), xy(4,4,2), u(4,4,9), p(4,4,5), d(4,4,5)
  real*8 :: h(4,4), hs(4,4), ht(4,4), gx(2), gs(2), gt(2), jac
  real*8 :: gu(9), us(9), ut(9), ux(9), uy(9), gp(5), ps(5), pt(5), gd(5)
  real*8 :: direct(4,4), history(4,4), step, ratio, mass, den, q, da, v, measure
  integer :: i,j,k,l,m,n
  read(*,*) step,ratio,mass
  read(*,*) sz
  read(*,*) xy
  read(*,*) u
  read(*,*) p
  read(*,*) d
  direct=0; history=0
  do m=1,n_gauss
    do n=1,n_gauss
      call basisfunctions(xgauss(m),xgauss(n),h,hs,ht)
      gx=0; gs=0; gt=0; gu=0; us=0; ut=0; gp=0; ps=0; pt=0; gd=0
      do i=1,4
        do j=1,4
          gx=gx+xy(i,j,:)*sz(i,j)*h(i,j)
          gs=gs+xy(i,j,:)*sz(i,j)*hs(i,j)
          gt=gt+xy(i,j,:)*sz(i,j)*ht(i,j)
          gu=gu+u(i,j,:)*sz(i,j)*h(i,j)
          us=us+u(i,j,:)*sz(i,j)*hs(i,j)
          ut=ut+u(i,j,:)*sz(i,j)*ht(i,j)
          gp=gp+p(i,j,:)*sz(i,j)*h(i,j)
          ps=ps+p(i,j,:)*sz(i,j)*hs(i,j)
          pt=pt+p(i,j,:)*sz(i,j)*ht(i,j)
          gd=gd+d(i,j,:)*sz(i,j)*h(i,j)
        enddo
      enddo
      jac=gs(1)*gt(2)-gt(1)*gs(2)
      ux=(gt(2)*us-gs(2)*ut)/jac
      uy=(-gt(1)*us+gs(1)*ut)/jac
      write(*,'(*(ES26.17E3,1X))') gx,gs,gt,gu,us,ut,ux,uy,gp,ps,pt,gd
      den=max(gp(5),.5d0)
      q=max(gp(4),0.d0)/den
      da=mass*(ratio*gd(4)-q*ratio*gd(5))/den
      do i=1,4
        do j=1,4
          v=h(i,j)*sz(i,j)
          measure=v*gx(1)*jac*wgauss(m)*wgauss(n)
          direct(i,j)=direct(i,j)-measure*step*(gp(1)+max(gp(2),0.d0))
          history(i,j)=history(i,j)-measure*da*gu(9)*gu(8)
        enddo
      enddo
    enddo
  enddo
  write(*,'(*(ES26.17E3,1X))') direct
  write(*,'(*(ES26.17E3,1X))') history
end program
