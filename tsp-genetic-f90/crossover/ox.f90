module ox
   use utils

   public :: order_crossover
   private :: order_crossover_child
contains
   subroutine order_crossover(parent1, parent2, rnd1, rnd2, child1, child2)
      integer, dimension(:), intent(in) :: parent1
      integer, dimension(size(parent1)), intent(in) :: parent2
      real :: rnd1, rnd2
      integer, dimension(size(parent1)), intent(out) :: child1, child2
      integer cut_start, cut_end, tmp

      cut_start = floor(rnd1 * size(parent1)) + 1
      cut_end = floor(rnd2 * size(parent1)) + 1
      if (cut_end < cut_start) then
         tmp = cut_start
         cut_start = cut_end
         cut_end = tmp
      end if

      child1(:) = order_crossover_child(parent1, parent2, cut_start, cut_end)
      child2(:) = order_crossover_child(parent2, parent1, cut_start, cut_end)
   end subroutine

   function order_crossover_child(parent1, parent2, cut_start, cut_end) result(child)
      integer, dimension(:) :: parent1
      integer, dimension(size(parent1)) :: parent2, iparent1, child
      integer cut_start, cut_end, target, child_idx, parent_size

      parent_size = size(parent2)
      child(cut_start:cut_end) = parent1(cut_start:cut_end)

      child_idx = wrap_to_top(cut_end + 1, parent_size)
      target = child_idx

      iparent1 = inverse_array(parent1)

      do while(child_idx .ne. cut_start)
         target = wrap_to_top(target, parent_size)
         if (in_bounds(cut_start, iparent1(parent2(target)), cut_end)) then
            target = target + 1
            cycle
         end if

         child(child_idx) = parent2(target)
         target = target + 1
         child_idx = wrap_to_top(child_idx + 1, parent_size)
      end do
   end function
end module ox
