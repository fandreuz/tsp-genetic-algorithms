module central_inverse
   public :: mutate
contains
   function mutate(chromosome, pos) result(mutated)
      integer, dimension(:) :: chromosome
      integer :: pos, L, i
      integer, dimension(size(chromosome)) :: mutated

      L = size(chromosome)
      do i = 1, pos
         mutated(i) = chromosome(pos - i + 1)
      end do
      do i = pos + 1, L
         mutated(i) = chromosome(pos + 1 + L - i)
      end do
   end function
end module central_inverse
