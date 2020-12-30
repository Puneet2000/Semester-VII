subroutine ge(a,b)
	implicit none
	real, dimension(3,3), intent(inout) :: a
	real, dimension(3), intent(inout) :: b
	real :: pivot, factor
	integer :: i, j, k, N
	N = 3

	do k = 2, N
	pivot = a(k-1,k-1)
		do i = k, N
		factor = a(i,k-1)/pivot
		b(i) = b(i) - factor * b(k-1)
			do j = 1, N
			a(i,j) = a(i,j) - factor * a(k-1,j)
			end do
	end do
	end do
end subroutine ge

subroutine backward_sweep(a,b,x)
	implicit none
	real, dimension(3,3), intent(in) :: a
	real,dimension(3), intent(in) :: b
	real,dimension(3), intent(out) :: x

	x(3) = b(3)/a(3,3)
	x(2) = (b(2) - (a(2,3) * x(3)))/a(2,2)
	x(1) = (b(1) - (a(1,3) * x(3)) - (a(1,2) * x(2)))/a(1,1)

end subroutine backward_sweep

program main
	implicit none

	real, dimension(3,3) :: a
	real, dimension(3) :: b, x
	a =  transpose(reshape((/ 5,-2,3,3,9,-5, 3, -2, 7 /), shape(a)))
	b = (/ 27, -11, 51 /)
	write(*,*) "A : ", a
	write(*,*) "b : ", b
	
	call ge(a,b)
	call backward_sweep(a,b,x)
	print*,"The values of x", x
end program main