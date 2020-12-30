subroutine ss(x,tolerance)
	implicit none
	real, dimension(2) :: tol, xold
	real, dimension(2), intent(inout) :: x
	real, intent(in) :: tolerance
	integer :: k,i
	do k = 0,100
		write (*,*) "Iteration number", k, " ", x
		xold = x
		x(1) = xold(1) + 2*xold(1)**2 - 5*xold(2)**3 -3
		x(2) = 3*xold(1)**3 + xold(2) + 2*xold(2)**2 -26

		do i = 1,2
		tol(i) = abs((x(i) - xold(i))/xold(i))
		end do
		if (tol(1) < tolerance .and. tol(2) < tolerance) then
			exit
		end if
	end do
end subroutine ss


program main
	implicit none

	real, dimension(2) :: x
	real :: tolerance
	x = (/ 1, 1 /)
	tolerance = 1e-5

	write(*,*) "Starting point: ", x
	
	call ss(x,tolerance)
end program main