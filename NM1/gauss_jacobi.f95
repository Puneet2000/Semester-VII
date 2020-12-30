subroutine jacobi(a,b,x,tolerance)
	implicit none
	real, dimension(3,3), intent(in) :: a
	real, dimension(3), intent(in) :: b
	real, dimension(3) :: tol, xold
	real, dimension(3), intent(inout) :: x
	real, intent(in) :: tolerance
	integer :: k,i
	do k = 0,100
		write (*,*) "Iteration number", k, " ", x
		xold = x
		x(1) = (b(1) - a(1,2) * xold(2) - a(1,3) * xold(3))/a(1,1)
		x(2) = (b(2) - a(2,1) * xold(1) - a(2,3) * xold(3))/a(2,2)
		x(3) = (b(3) - a(3,1) * xold(1) - a(3,2) * xold(2))/a(3,3)

		do i = 1,3
		tol(i) = abs((x(i) - xold(i))/xold(i))
		end do
		if (tol(1) < tolerance .and. tol(2) < tolerance .and. tol(3) < tolerance) then
			exit
		end if
	end do
end subroutine jacobi


program main
	implicit none

	real, dimension(3,3) :: a
	real, dimension(3) :: b, x
	real :: tolerance
	a =  transpose(reshape((/ 5,-2,3,3,9,-5, 3, -2, 7 /), shape(a)))
	b = (/ 27, -11, 51 /)
	x = (/ 1, 2, 1 /)
	tolerance = 1e-5

	write(*,*) "A : ", a
	write(*,*) "b : ", b
	write(*,*) "Starting point: ", x
	
	call jacobi(a,b,x,tolerance)
end program main