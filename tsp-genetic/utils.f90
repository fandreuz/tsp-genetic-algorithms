module utils
contains
   function inverse_array(array) result(inverted)
      integer, dimension(:) :: array
      integer, dimension(size(array)) :: inverted

      do i=1, size(array)
         ! assume the smallest value is one
         inverted(array(i)) = i
      end do
   end function

   function out_of_bounds(small, x, big) result(result)
      integer :: small, x, big
      logical :: result

      result = x < small .or. big < x
   end function

   ! more efficient than negating out_of_bounds
   function in_bounds(small, x, big) result(result)
      integer :: small, x, big
      logical :: result

      result = small <= x .and. x <= big
   end function
end module utils
