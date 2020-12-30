subroutine my_jacobi(x,jacobi)
	implicit none

	real,dimension(2),intent(in) :: x
	real,dimension(2,2),intent(out) :: jacobi

	jacobi(1,1) = 4*x(1)
	jacobi(1,2) = -15*x(2)**2
	jacobi(2,1) = 9*x(1)**2
	jacobi(2,2) = 4*x(2)

end subroutine my_jacobi

subroutine getInverse2D(a,aInverse)
	implicit none

	real,dimension(2,2),intent(in) :: a
	real,dimension(2,2),intent(out) :: aInverse
	real :: detA
	aInverse(1,1) = a(2,2)
	aInverse(1,2) = - a(1,2)
	aInverse(2,1) = - a(2,1)
	aInverse(2,2) = a(1,1)
	detA = (a(1,1) * a(2,2)) - (a(1,2) * a(2,1))

	aInverse = aInverse/detA

	end subroutine getInverse2D

subroutine my_fun(x,f)

	implicit none
	real,dimension(2),intent(in) :: x
	real,dimension(2),intent(out) :: f

	f(1) = 2*x(1)**2 - 5*x(2)**3 -3
	f(2) = 3*x(1)**3 + 2*x(2)**2 -26

end subroutine my_fun

subroutine matrixProd(a,b,c)
	implicit none
	real, dimension(2,2),intent(in) :: a
	real, dimension(2,1), intent(in) :: b
	real, dimension(2,1), intent(out) :: c
	integer :: ar, ac, br = 1, bc = 1
	real :: summation = 0
	do ar = 1,2
		do ac = 1,2
			c(ar, bc) = summation + a(ar, ac) * b(br, bc)
			summation = c(ar, bc)
			br = br + 1
		end do
		br = 1
		summation = 0
	end do
end subroutine matrixProd

program main
	implicit none
	real, dimension(2) :: x, xPrevious,prodMat, f
	real, dimension(2,2) :: jacobi, jacobinverse
	real :: tol = 1.0e-5
	integer :: i

	print *, "Enter initial guess for x: "
	read *, x
	xPrevious(1) = x(1)
	xPrevious(2) = x(2)
	do i = 1,100
		call my_fun(x,f)
		call my_jacobi(x,jacobi)
		call getInverse2D(jacobi, jacobinverse)
		call matrixProd(jacobinverse, f, prodMat)
		write(*, *) i, x(1), x(2), f(1), f(2)

		x(1) = xPrevious(1) - prodMat(1)
		x(2) = xPrevious(2) - prodMat(2)

		if ((abs((x(1) - xPrevious(1))/xPrevious(1)) .le. tol) .and. (abs((x(2) - xPrevious(2))/xPrevious(2)) .le. tol)) then
			exit
		end if
		xPrevious = x
	end do
	print *, "The solution is x1 = ", x(1), " x2 = ", x(2)
end program main