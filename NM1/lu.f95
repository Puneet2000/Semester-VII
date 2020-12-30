subroutine display(a)
    implicit none
    real, dimension(3,3), intent(in) :: a
    print *, a(1,1), a(1,2), a(1,3)
    print *, a(2,1), a(2,2), a(2,3)
    print *, a(3,1), a(3,2), a(3,3)
end subroutine display

subroutine doolittle(a, l, u)
    implicit none
    real, dimension(3,3), intent(inout) :: a, l, u
    integer :: i, j, k, N
    real :: sum
    N = 3
    l =  transpose(reshape((/ 0,0,0,0,0,0,0,0,0 /), shape(l)))
    u =  transpose(reshape((/ 0,0,0,0,0,0,0,0,0 /), shape(u)))

    do i = 1, N
        do k = i, N
            sum = 0
            do j = 1,i
                sum = sum + l(i,j)*u(j,k)
            end do
            u(i,k) = a(i,k) - sum

        end do

        do k = i, N
            if (i==k) then
                l(i,i) = 1
            else
                sum = 0
                do j = 1,i
                    sum = sum + l(k,j)*u(j,i)
                end do
                l(k,i) = (a(k,i) - sum)/u(i,i)
            end if
         end do

    end do
end subroutine doolittle

subroutine backward_sweep(a,b,x)
    implicit none
    real, dimension(3,3), intent(in) :: a
    real,dimension(3), intent(in) :: b
    real,dimension(3), intent(out) :: x

    x(3) = b(3)/a(3,3)  ! backward sweep from x3 to x1
    x(2) = (b(2) - (a(2,3) * x(3)))/a(2,2)
    x(1) = (b(1) - (a(1,3) * x(3)) - (a(1,2) * x(2)))/a(1,1)

end subroutine backward_sweep

subroutine forward_sweep(a,b,x)
    implicit none
    real, dimension(3,3), intent(in) :: a
    real,dimension(3), intent(in) :: b
    real,dimension(3), intent(out) :: x

    x(1) = b(1)/a(1,1)  ! backward sweep from x3 to x1
    x(2) = (b(2) - (a(2,1) * x(1)))/a(2,2)
    x(3) = (b(3) - (a(3,1) * x(1)) - (a(3,2) * x(2)))/a(3,3)
end subroutine forward_sweep

program main
    implicit none

    real, dimension(3,3) :: a, l, u ! A, L, U matrix
    real, dimension(3) :: b, x ! b vector
    a =  transpose(reshape((/ 5,-2,3,3,9,-5, 3, -2, 7 /), shape(a)))
    b = (/ 27, -11, 51 /)
    print *, "A : "
    call display(a) ! pretty print matrix A
    write(*,*) "b : ", b

    call doolittle(a,l,u) ! doing gauss elimination
    print *, "L : "
    call display(l)
    print *, "U : "
    call display(u)

    call forward_sweep(l,b,x)
    print*,"The values of x", x ! print values
end program main
