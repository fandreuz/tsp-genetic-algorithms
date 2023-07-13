module optimum_search
   use utils

   public :: fitness
   private :: decrease_if_above
contains
   function fitness(cost_matrix, population) result(value)
      real, dimension(:,:) :: cost_matrix
      integer, dimension(:, :) :: population
      real, dimension(size(population, 1)) :: value
      integer :: i, j, target_column

      value(:) = 0.0
      do i=1, size(population, 1)
         do j=1, size(cost_matrix, 1) - 1
            target_column = decrease_if_above(population(i, j + 1), population(i, j))
            value(i) = value(i) + cost_matrix(population(i, j), target_column)
         end do
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
end module optimum_search


