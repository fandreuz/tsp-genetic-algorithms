module cx
   use utils

   public :: cycle_crossover
   private :: cycle_crossover_child
contains
   subroutine cycle_crossover(parent1, parent2, child1, child2)
      integer, dimension(:), intent(in) :: parent1
      integer, dimension(size(parent1)), intent(in) :: parent2
      integer, dimension(size(parent1)), intent(out) :: child1, child2

      child1(:) = cycle_crossover_child(parent1, parent2)
      child2(:) = cycle_crossover_child(parent2, parent1)
   end subroutine

   function cycle_crossover_child(parent1, parent2) result(child)
      integer, dimension(:) :: parent1
      integer, dimension(size(parent1)) :: parent2
      integer, dimension(size(parent1)) :: child
      logical, dimension(size(child)) :: child_bitmap
      integer, dimension(size(parent1)) :: iparent1
      integer target

      child_bitmap(:) = .false.
      child(:) = parent2(:)

      child(1) = parent1(1)
      child_bitmap(1) = .true.

      iparent1 = inverse_array(parent1)

      target = iparent1(parent2(1))
      do while (.not. child_bitmap(target))
         child(target) = parent1(target)
         child_bitmap(target) = .true.
         target = iparent1(parent2(target))
      end do
   end function
end module cx

