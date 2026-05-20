module force_average
    !
    ! This module initialises and finalises the time-averaging objects for the
    ! radial, latitudinal and azimuthal components of the Lorentz force.
    ! The averaging itself is handled by the calc_ave module.
    !
    use calc_ave
    use radial_data,  only: nRstart, nRstop
    use truncation,   only: n_phi_max, nlat_padded
    implicit none

    type(ave), public :: LFr_ave  ! Time-averaged radial Lorentz force
    type(ave), public :: LFt_ave  ! Time-averaged colatitudinal Lorentz force
    type(ave), public :: LFp_ave  ! Time-averaged azimuthal Lorentz force

    public :: initialize_force_average, finalize_force_average

contains

    subroutine initialize_force_average
        !
        ! Allocate and initialise the averaging objects for all three
        ! Lorentz force components on the local radial subdomain.
        !
        call LFr_ave%init(nRstart, nRstop, nlat_padded, n_phi_max)
        call LFt_ave%init(nRstart, nRstop, nlat_padded, n_phi_max)
        call LFp_ave%init(nRstart, nRstop, nlat_padded, n_phi_max)

    end subroutine initialize_force_average


    subroutine finalize_force_average
        !
        ! Deallocate the averaging objects for all three Lorentz force components.
        !
        call LFr_ave%finalize()
        call LFt_ave%finalize()
        call LFp_ave%finalize()

    end subroutine finalize_force_average

end module force_average