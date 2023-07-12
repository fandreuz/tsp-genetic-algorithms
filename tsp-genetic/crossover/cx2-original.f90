module cx2_original
   public :: cycle_crossover2
contains
   subroutine cycle_crossover2(parent1, parent2, child1, child2)
      integer, dimension(:), intent(in) :: parent1
      integer, dimension(size(parent1)), intent(in) :: parent2
      integer, dimension(size(parent1)), intent(out) :: child1, child2
      logical, dimension(size(parent1)) :: parent_bitmap
      integer :: target, child_idx, ind1(1), ind2(1)

      parent_bitmap(:) = .false.
      target = 1
      do child_idx=child_idx, size(parent1)
         if (parent_bitmap(target)) then
            do target=1, size(parent1)
               if (.not. parent_bitmap(target)) then
                  exit
               end if
            end do
         end if

         child1(child_idx) = parent2(target)

         ind1 = findloc(parent1, child1(child_idx))
         ind2 = findloc(parent1, parent2(ind1(1)))
         child2(child_idx) = parent2(ind2(1))
         parent_bitmap(target) = .true.

         ind1 = findloc(parent1, child2(child_idx))
         target = ind1(1)
      end do
   end subroutine
end module cx2_original
