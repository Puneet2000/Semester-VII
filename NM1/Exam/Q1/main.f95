subroutine my_jacobi(x,jacobi)
    implicit none

    real,dimension(2),intent(in) :: x
    real,dimension(2,2),intent(out) :: jacobi

    ! calculating Jacobian [[ 4x_1,  -15x_2], [9x_1, 4x_2]]
    jacobi(1,1) = -SIN(x(1) + x(2)) + 2
    jacobi(1,2) = -SIN(x(1) + x(2)) - 2
    jacobi(2,1) = -SIN(x(1) + x(2)) - 2
    jacobi(2,2) = -SIN(x(1) + x(2)) + 2

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

    f(1) = COS(x(1) + x(2)) + 2*(x(1)-x(2)) -1.5
    f(2) = COS(x(1) + x(2)) - 2*(x(1)-x(2)) + 2.5

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

subroutine obj(x,f)

    implicit none
    real,dimension(2),intent(in) :: x
    real,intent(out) :: f

    f = SIN(x(1) + x(2)) + (x(1)-x(2))**2 - 1.5*x(1) + 2.5*x(2) + 1

end subroutine obj


program main
    implicit none
    real, dimension(2) :: x, xPrevious,prodMat, f
    real, dimension(2,2) :: jacobi, jacobinverse
    real :: tol = 1.0e-7, func
    integer :: i

    print *, "Solving: cos(x1+x2) + 2(x1-x2) -1.5 = 0 and cos(x1+x2) - 2(x1-x2) + 2.5  = 0" ! equations to solve

    x = (/0 , -2 /)
    xPrevious = (/ 0, -2 /) ! Runnning algorithm for value (0,-2)
    do i = 1,100
        call my_fun(x,f) ! get f(x)
        call my_jacobi(x,jacobi) ! get A(x)
        call getInverse2D(jacobi, jacobinverse) ! get A^-1(x)
        call matrixProd(jacobinverse, f, prodMat) ! get A^-1(x). f(x)
        write(*, *) i, x(1), x(2)

        x(1) = xPrevious(1) - prodMat(1) ! update values
        x(2) = xPrevious(2) - prodMat(2)

        if ((abs((x(1) - xPrevious(1))/xPrevious(1)) .le. tol) .and. (abs((x(2) - xPrevious(2))/xPrevious(2)) .le. tol)) then
            exit ! check convergence, terminate if acheived
        end if
        xPrevious = x
    end do
    print *, "The solution is x1 = ", x(1), " x2 = ", x(2)
    call obj(x,func)
    print *, "The value of f(x) at optimal x1 and x2 is", func
end program main
