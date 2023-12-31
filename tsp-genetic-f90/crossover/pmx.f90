module pmx
   use utils

   public :: partially_mapped_crossover
   private :: partially_mapped_crossover_child
contains
   subroutine partially_mapped_crossover(parent1, parent2, pos1, pos2, child1, child2)
      integer, dimension(:), intent(in) :: parent1
      integer, dimension(size(parent1)), intent(in) :: parent2
      integer, intent(in) :: pos1, pos2
      integer, dimension(size(parent1)), intent(out) :: child1, child2
      integer, dimension(size(parent1)) :: iparent1, iparent2
      integer :: cut_start, cut_end

      if (pos1 <= pos2) then
         cut_start = pos1
         cut_end = pos2
      else
         cut_start = pos2
         cut_end = pos1
      end if

      iparent1 = inverse_array(parent1)
      iparent2 = inverse_array(parent2)

      child1(:) = partially_mapped_crossover_child(parent1, parent2, iparent1, iparent2, cut_start, cut_end)
      child2(:) = partially_mapped_crossover_child(parent2, parent1, iparent2, iparent1, cut_start, cut_end)
   end subroutine

   function partially_mapped_crossover_child(parent1, parent2, iparent1, iparent2, cut_start, cut_end) result(child)
      integer, dimension(:) :: parent1
      integer, dimension(size(parent1)) :: parent2, child, iparent1, iparent2
      integer cut_start, cut_end, target, i

      child(:) = parent2(:)
      child(cut_start:cut_end) = parent1(cut_start:cut_end)
      do i=cut_start, cut_end
         target = iparent2(child(i))
         if (out_of_bounds(cut_start, target, cut_end)) then
            do while (in_bounds(cut_start, iparent1(child(target)), cut_end))
               child(target) = parent2(iparent1(child(target)))
            end do
         end if
      end do
   end function
end module pmx
