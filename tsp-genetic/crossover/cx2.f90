module cx2
   use utils

   public :: cycle_crossover2
contains
   subroutine cycle_crossover2(parent1, parent2, child1, child2)
      integer, dimension(:), intent(in) :: parent1
      integer, dimension(size(parent1)), intent(in) :: parent2
      integer, dimension(size(parent1)), intent(out) :: child1, child2
      integer, dimension(size(parent1)) :: iparent1, iparent2
      logical, dimension(size(parent1)) :: parent_bitmap
      integer :: target, child_idx

      iparent1 = inverse_array(parent1)
      iparent2 = inverse_array(parent2)

      parent_bitmap(:) = .false.
      target = 1
      do child_idx=1, size(parent1)
         if (parent_bitmap(target)) then
            do target=1, size(parent1)
               if (.not. parent_bitmap(target)) then
                  exit
               end if
            end do
         end if

         child1(child_idx) = parent2(target)
         child2(child_idx) = parent2(iparent1(parent2(iparent1(child1(child_idx)))))
         parent_bitmap(target) = .true.

         target = iparent1(child2(child_idx))
      end do
   end subroutine
end module cx2

