module optimum_search
   use utils
   use configuration

   public :: find_optimum
   private :: initialize_population
contains
   function find_optimum(cost_matrix) result(optimum)
      real, dimension(:,:) :: cost_matrix
      real, dimension(size(cost_matrix, 1)) :: optimum
      integer, dimension(population_size, size(cost_matrix, 1)) :: population

      call initialize_population(population)
   end function

   subroutine initialize_population(population)
      integer, dimension(:, :), intent(out) :: population
      integer :: i

      do i = 1, population_size
         ! subroutine becomes a function due to f2py
         population(i, :) = scramble(size(population, 2))
      end do
   end subroutine
end module optimum_search


