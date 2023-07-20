module reverse_sequence
   public :: mutate
contains
   function mutate(chromosome, pos1, pos2) result(mutated)
      integer, dimension(:) :: chromosome
      integer :: pos1, pos2, i, cut_start, cut_end
      integer, dimension(size(chromosome)) :: mutated

      if (pos1 <= pos2) then
         cut_start = pos1
         cut_end = pos2
      else
         cut_start = pos2
         cut_end = pos1
      end if

      mutated(:) = chromosome(:)
      do i = cut_start, cut_end
         mutated(i) = chromosome(cut_end - i + cut_start)
      end do
   end function
end module reverse_sequence