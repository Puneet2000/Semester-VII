subroutine sor(a,b,x,tolerance, w)
	implicit none
	real, dimension(3,3), intent(in) :: a
	real, dimension(3), intent(in) :: b
	real, dimension(3) :: tol, xold
	real, dimension(3), intent(inout) :: x
	real, intent(in) :: tolerance, w
	integer :: k,i
	do k = 0,100
		write (*,*) "Iteration number", k, " ", x
		xold = x
		x(1) = (1-w)*xold(1) + w*(b(1) - a(1,2) * x(2) - a(1,3) * x(3))/a(1,1)
		x(2) = (1-w)*xold(2) + w*(b(2) - a(2,1) * x(1) - a(2,3) * x(3))/a(2,2)
		x(3) = (1-w)*xold(3) + w*(b(3) - a(3,1) * x(1) - a(3,2) * x(2))/a(3,3)

		do i = 1,3
		tol(i) = abs((x(i) - xold(i))/xold(i))
		end do
		if (tol(1) < tolerance .and. tol(2) < tolerance .and. tol(3) < tolerance) then
			exit
		end if
	end do
end subroutine sor


program main
	implicit none

	real, dimension(3,3) :: a
	real, dimension(3) :: b, x
	real :: tolerance, weight
	a =  transpose(reshape((/ 5,-2,3,3,9,-5, 3, -2, 7 /), shape(a)))
	b = (/ 27, -11, 51 /)
	x = (/ 1, 2, 1 /)
	tolerance = 1e-5
	weight = 1.5

	write(*,*) "A : ", a
	write(*,*) "b : ", b
	write(*,*) "Starting point: ", x
	write(*,*) "Weight w: ", weight
	
	call sor(a,b,x,tolerance, weight)
end program main