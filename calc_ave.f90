module calc_ave
    !
    ! This module computes time-averaged field quantities such as the Lorentz force.
    ! Averaging is performed per radial level, using a weighted running
    ! average that accounts for variable time steps.
    !
    use precision_mod
    use constants, only: zero
    implicit none
    private

    type, public :: ave
        real(cp), allocatable :: f_ave(:,:,:)  ! Time-averaged field, shape (nr, nt, np)
        real(cp), allocatable :: time_last(:)  ! Time of the last call at each radial level
        real(cp), allocatable :: time_ave(:)   ! Total time elapsed since averaging began
        integer,  allocatable :: nTimes(:)     ! Number of times add_r has been called
        integer :: nt, np
    contains
        procedure :: init
        procedure :: add_r
        procedure :: finalize
    end type ave

contains

    subroutine init(this, nr_start, nr_stop, nt_I, np_I)
        !
        ! Allocate and initialise the averaging arrays for the given grid dimensions.
        !
        class(ave),  intent(inout) :: this
        integer,     intent(in)    :: nr_start, nr_stop  ! Radial index range
        integer,     intent(in)    :: nt_I, np_I         ! Colatitudinal and azimuthal grid sizes

        allocate(this%f_ave(nr_start:nr_stop, nt_I, np_I))
        allocate(this%nTimes(nr_start:nr_stop))
        allocate(this%time_last(nr_start:nr_stop))
        allocate(this%time_ave(nr_start:nr_stop))

        this%f_ave    = 0.0_cp
        this%nTimes   = 0
        this%time_last= 0.0_cp
        this%time_ave = 0.0_cp
        this%nt       = nt_I
        this%np       = np_I

    end subroutine init


    subroutine add_r(this, f_in, time_in, nR)
        !
        ! Update the running time-average at radial level nR with the new field f_in.
        !
        ! On the first call, the average is set to f_in. On subsequent calls, a
        ! time-weighted average is computed:
        !
        !   f_ave_new = (f_ave_old * time_ave + f_in * dt) / (time_ave + dt)
        !
        class(ave),  intent(inout) :: this
        real(cp),    intent(in)    :: f_in(:,:)  ! New field snapshot, shape (nt, np)
        real(cp),    intent(in)    :: time_in    ! Simulation time of the new snapshot
        integer,     intent(in)    :: nR         ! Radial level index
        real(cp) :: dt

        if ( size(f_in, 1) /= this%nt .or. size(f_in, 2) /= this%np ) then
            stop "calc_ave: add_r: input field dimensions do not match initialised grid"
        end if

        this%nTimes(nR) = this%nTimes(nR) + 1

        if ( this%nTimes(nR) == 1 ) then
            ! First call: initialise average to the first snapshot
            this%f_ave(nR,:,:) = f_in
        else
            dt = time_in - this%time_last(nR)
            this%f_ave(nR,:,:) = ( this%f_ave(nR,:,:) * this%time_ave(nR) &
                                 + f_in * dt ) / ( this%time_ave(nR) + dt )
            this%time_ave(nR)  = this%time_ave(nR) + dt
        end if

        this%time_last(nR) = time_in

    end subroutine add_r


    subroutine finalize(this)
        !
        ! Deallocate all used arrays.
        !
        class(ave), intent(inout) :: this

        deallocate(this%f_ave)
        deallocate(this%nTimes)
        deallocate(this%time_last)
        deallocate(this%time_ave)

    end subroutine finalize

end module calc_ave