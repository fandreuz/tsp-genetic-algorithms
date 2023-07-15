module reverse_sequence
   public :: mutate
contains
   function mutate(chromosome, pos1, pos2) result(mutated)
      integer, dimension(:) :: chromosome
      integer :: pos1, pos2, i
      integer, dimension(size(chromosome)) :: mutated

      mutated(:) = chromosome(:)
      do i = pos1, pos2
         mutated(i) = chromosome(pos2 - i + pos1)
      end do
   end function
end module reverse_sequence