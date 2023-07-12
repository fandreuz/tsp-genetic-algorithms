module swap_mutation
   public :: swap
contains
   function swap(chromosome, pos1, pos2) result(mutated)
      integer, dimension(:) :: chromosome
      integer :: pos1, pos2
      integer, dimension(size(chromosome)) :: mutated

      mutated(:) = chromosome(:)
      mutated(pos1) = chromosome(pos2)
      mutated(pos2) = chromosome(pos1)
   end function
end module swap_mutation
