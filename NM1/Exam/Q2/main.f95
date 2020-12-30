subroutine my_grad(x,grad)
    implicit none

    real,dimension(2),intent(in) :: x
    real,dimension(2),intent(out) :: grad

    grad(1) = COS(x(1)+x(2)) + 2*(x(1)-x(2)) -1.5
    grad(2) = COS(x(1)+x(2)) - 2*(x(1)-x(2)) + 2.5

end subroutine my_grad

subroutine obj(x,f)

    implicit none
    real,dimension(2),intent(in) :: x
    real,intent(out) :: f

    f = SIN(x(1) + x(2)) + (x(1)-x(2))**2 - 1.5*x(1) + 2.5*x(2) + 1

end subroutine obj

program main
    implicit none
    real, dimension(2) :: x, xPrevious,grad
    real :: step_size = 0.001
    real :: tol = 1.0e-7, func
    integer :: i

    print *, "Solving: cos(x1+x2) + 2(x1-x2) -1.5 = 0 and cos(x1+x2) - 2(x1-x2) + 2.5  = 0" ! equations to solve

    x = (/0 , -2 /)
    xPrevious = (/ 0, -2 /) ! Runnning algorithm for value (0,-2)
    do i = 1,1000
        call my_grad(xPrevious, grad)
        x(1) = xPrevious(1) - step_size*grad(1)
        x(2) = xPrevious(2) - step_size*grad(2)
        write(*, *) i, x(1), x(2)

        if ((abs((x(1) - xPrevious(1))/xPrevious(1)) .le. tol) .and. (abs((x(2) - xPrevious(2))/xPrevious(2)) .le. tol)) then
            exit ! check convergence, terminate if acheived
        end if
        xPrevious = x
    end do
    print *, "The solution is x1 = ", x(1), " x2 = ", x(2)

    call obj(x,func)
    print *, "The value of f(x) at optimal x1 and x2 is", func
end program main
