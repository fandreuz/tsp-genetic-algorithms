module optimum_search
   use utils
   use configuration

   public :: find_optimum
   private :: initialize_population, fitness, random_permutation
contains
   function find_optimum(cost_matrix) result(optimum)
      real, dimension(:,:) :: cost_matrix
      real, dimension(size(cost_matrix, 1)) :: optimum
      integer, dimension(population_size, size(cost_matrix, 1)) :: population

      call initialize_population(population)
      write(*,*) random_permutation(size(population, 1))
   end function

   subroutine initialize_population(population)
      integer, dimension(:, :), intent(out) :: population
      integer :: i

      do i = 1, population_size
         ! subroutine becomes a function due to f2py
         population(i, :) = random_permutation(size(population, 1))
      end do
   end subroutine

   ! inspired by programming-idioms.org, slightly modified
   function random_permutation(n_values) result(array)
      integer :: n_values, k, i, j, itemp
      integer, dimension(n_values) :: array
      real, dimension(2 * n_values) :: rnd

      do i = 1, n_values
         array(i) = i
      end do

      call random_number(rnd)
      do k = 1, 2
         do i = 1, n_values
            j = 1 + floor(n_values * rnd((k - 1) * n_values + i))
            itemp = array(j)
            array(j) = array(i)
            array(i) = itemp
         enddo
      enddo
   end function random_permutation

   function decrease_if_above(x, top) result(y)
      integer :: x, top, y

      if (x < top) then
         y = x
      else
         y = x - 1
      end if
   end function

   function fitness(cost_matrix, population) result(value)
      real, dimension(:,:) :: cost_matrix
      integer, dimension(population_size, size(cost_matrix, 1)) :: population
      real, dimension(population_size) :: value
      integer :: i, j, target_column

      value(:) = 0.0
      do i=1, population_size
         do j=1, size(cost_matrix, 1) - 1
            target_column = decrease_if_above(population(i, j + 1), population(i, j))
            value(i) = value(i) + cost_matrix(population(i, j), target_column)
         end do
      end do
   end function
end module optimum_search


