module problem
   use utils

   public :: fitness
   private :: decrease_if_above
contains
   function fitness(cost_matrix, population) result(value)
      real, dimension(:,:) :: cost_matrix
      integer, dimension(:,:) :: population
      real, dimension(size(population, 1)) :: value
      integer :: i, j, current, next

      value(:) = 0.0
      do i=1, size(population, 1)
         current = population(i, 1)
         do j=2, size(cost_matrix, 1)
            next = population(i, j)
            value(i) = value(i) + cost_matrix(current, decrease_if_above(next, current))
            current = next
         end do

         ! back where you belong!
         next = population(i, 1)
         value(i) = value(i) + cost_matrix(current, decrease_if_above(next, current))
      end do
   end function

   function decrease_if_above(x, top) result(y)
      integer :: x, top, y

      if (x < top) then
         y = x
      else
         y = x - 1
      end if
   end function
end module problem


