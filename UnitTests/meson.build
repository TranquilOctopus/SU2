# Add any new test files here
# -------------------------------------------------------------------------
# Begin unit test listings
# -------------------------------------------------------------------------

# Direct-mode tests:
su2_cfd_tests = files(['Common/CConfig_tests.cpp',
                       'Common/geometry/primal_grid/CPrimalGrid_tests.cpp',
                       'Common/geometry/dual_grid/CDualGrid_tests.cpp',
                       'Common/geometry/CGeometry_test.cpp',
                       'Common/toolboxes/CQuasiNewtonInvLeastSquares_tests.cpp',
                       'Common/toolboxes/C1DInterpolation_tests.cpp',
                       'Common/vectorization.cpp',
                       'Common/toolboxes/ndflattener_tests.cpp',
                       'Common/containers/CLookupTable_tests.cpp',
                       'SU2_CFD/fluid/CFluidCantera_tests.cpp',
                       'Common/toolboxes/multilayer_perceptron/CLookUp_ANN_tests.cpp',
                       'SU2_CFD/numerics/CNumerics_tests.cpp',
                       'SU2_CFD/fluid/CFluidModel_tests.cpp',
                       'SU2_CFD/output/CNEMOCompOutput_tests.cpp',
                       'SU2_CFD/output/COutput_convergence_tests.cpp',
                       'SU2_CFD/gradients.cpp',
                       'SU2_CFD/periodic_limiters.cpp',
                       'SU2_CFD/solvers/CNEMOEulerSolver_tests.cpp',
                       'SU2_CFD/nemo_viscous_assembly.cpp',
                       'SU2_CFD/windowing.cpp',
                       'Common/toolboxes/random_toolbox_tests.cpp',
                       'Common/linear_algebra/quantization_tests.cpp',
                       'Common/linear_algebra/edge_residual_blocks_tests.cpp',
                       'Common/linear_algebra/CSysGPU_tests.cpp'])

# Reverse-mode (algorithmic differentiation) tests:
su2_cfd_tests_ad = files(['Common/simple_ad_test.cpp',
                          'SU2_CFD/solvers/under_relaxation_ad_tests.cpp',
                          'SU2_CFD/fluid/CFluidModel_tests_AD.cpp',
                          'Common/toolboxes/multilayer_perceptron/MLP_Jacobian_tests.cpp'])

# Forward-mode (direct differentiation) tests:
su2_cfd_tests_dd = files(['Common/simple_directdiff_test.cpp',
                          'SU2_CFD/solvers/under_relaxation_ad_tests.cpp'])

# Tests of the CUDA kernels, they need a device. The .cu file is built by nvcc, which takes only
# the defines out of su2_cpp_args (see common_cuda_src in Common/src/meson.build). The stale data
# checks of the device mirrors are switched on for these tests.
su2_cuda_tests = []
su2_cuda_test_args = []
su2_cuda_test_cuda_args = []
if get_option('enable-cuda')
  su2_cuda_tests = files(['Common/containers/CDeviceMirror_tests.cpp',
                          'Common/containers/CDeviceMirror_kernels.cu'])
  su2_cuda_test_args = ['-DSU2_GPU_CHECK_STALE']
  foreach arg : su2_cpp_args
    if arg.startswith('-D') or arg.startswith('-U')
      su2_cuda_test_cuda_args += arg
    endif
  endforeach
  su2_cuda_test_cuda_args += '-DSU2_GPU_CHECK_STALE'
endif

# -------------------------------------------------------------------------
# End of unit test listings
# -------------------------------------------------------------------------


if get_option('enable-tests')
  if get_option('enable-normal')
    unit_test_files = su2_cfd_tests + su2_cuda_tests + files(['test_driver.cpp'])
    dependencies_default = [su2_cfd_dep, common_dep, su2_deps, catch2_dep]
    test_driver = executable(
        'test_driver',
        unit_test_files,
        install : true,
        install_rpath : '$ORIGIN/../lib',
        dependencies : dependencies_default,
        cpp_args: ['-fPIC', default_warning_flags, su2_cpp_args, su2_cuda_test_args],
        cuda_args: su2_cuda_test_cuda_args
    )
    test('Catch2 test driver', test_driver)
    test('NEMO convergence field migration', python,
         args: files('SU2_CFD/output/check_nemo_convergence_migration.py') + [test_driver.full_path()],
         timeout: 300)
  endif

  if get_option('enable-autodiff')
    unit_test_files_ad = su2_cfd_tests_ad + files(['test_driver.cpp'])
    test_driver_AD = executable(
        'test_driver_AD',
        unit_test_files_ad,
        install : true,
        install_rpath : '$ORIGIN/../lib',
        dependencies : [su2_cfd_dep_ad, commonAD_dep, su2_deps, codi_dep, catch2_dep],
        cpp_args: ['-fPIC', default_warning_flags, su2_cpp_args, codi_rev_args]
    )
    test('Catch2 test driver (AD)', test_driver_AD)
  endif

  if get_option('enable-directdiff')
    unit_test_files_dd = su2_cfd_tests_dd + files(['test_driver.cpp'])
    test_driver_DD = executable(
        'test_driver_DD',
        unit_test_files_dd,
        install : true,
        install_rpath : '$ORIGIN/../lib',
        dependencies : [su2_cfd_dep_dd, commonDD_dep, su2_deps, codi_dep, catch2_dep],
        cpp_args: ['-fPIC', default_warning_flags, su2_cpp_args, codi_for_args]
    )
    test('Catch2 test driver (DD)', test_driver_DD)
  endif
endif
