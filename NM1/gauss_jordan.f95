subroutine gje(a,b, Inv)
	implicit none
	real, dimension(3,3), intent(inout) :: a, Inv
	real, dimension(3), intent(inout) :: b
	real :: pivot, factor
	integer :: i, j, k, N
	Inv =  transpose(reshape((/ 1,0,0,0,1,0, 0, 0, 1 /), shape(Inv)))
	N = 3

	do k = 2, N
	pivot = a(k-1,k-1)
		do i = k, N
		factor = a(i,k-1)/pivot
		b(i) = b(i) - factor * b(k-1)
			do j = 1, N
			a(i,j) = a(i,j) - factor * a(k-1,j)
			Inv(i,j) = Inv(i,j) - factor * Inv(k-1,j)
			end do
		end do
	end do

	! print *, a

	do k = 2, N
	pivot = a(N-k+2,N-k+2)
		do i = k, N
		factor = a(N-i+1,N-k+2)/pivot
		b(N-i+1) = b(N-i+1) - factor * b(N-k+2)
			do j = 1, N
			a(N-i+1,N-j+1) = a(N-i+1,N-j+1) - factor * a(N-k+2,N-j+1)
			Inv(N-i+1,N-j+1) = Inv(N-i+1,N-j+1) - factor * Inv(N-k+2,N-j+1)
			end do
		end do
	end do

	! print *, a

	do k = 1, N
	factor = a(k,k)
	b(k) = b(k)/factor
		do i = 1, N
		a(k,i) = a(k,i)/factor
		Inv(k,i) = Inv(k,i)/factor
		end do
	end do

	! print *, a

end subroutine gje

program main
	implicit none

	real, dimension(3,3) :: a, I
	real, dimension(3) :: b
	a =  transpose(reshape((/ 5,-2,3,3,9,-5, 3, -2, 7 /), shape(a)))
	b = (/ 27, -11, 51 /)
	write(*,*) "A : ", a
	write(*,*) "b : ", b
	
	call gje(a,b, I)
	print*,"A", a
	print*,"A inverse", I
	print*,"The values of x", b
end program main