common_src += files(['CGeometry.cpp',
                     'CPhysicalGeometry.cpp',
                     'CPhysicalGeometryFEM.cpp',
                     'CMultiGridGeometry.cpp',
                     'CDummyGeometry.cpp',
                     'CMultiGridQueue.cpp',
                     'CGeometryDevice.cpp',
                     'CDeviceHaloExchange.cpp'])

if get_option('enable-cuda')
  # Kept apart from common_src because nvcc takes a different set of flags, see
  # common_cuda_src in Common/src/meson.build.
  common_cuda_src += files(['CDeviceHaloExchangeGPU.cu'])
endif
