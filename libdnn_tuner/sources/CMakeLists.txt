cmake_minimum_required(VERSION 2.8.7)
if(POLICY CMP0046)
    cmake_policy(SET CMP0046 NEW)
endif()
if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()


message(INFO ${CMAKE_CXX_COMPILER})

SET(CMAKE_CXX_FLAGS "-std=c++11")


include_directories(/home/zrji/android_caffe/tmp_ndk/android-ndk-r18b/sysroot/usr/include/aarch64-linux-android)

#set(log-lib /home/zrji/android_caffe/tmp_ndk/android-ndk-r18b/platforms/android-28/arch-arm64/usr/lib/liblog.so)

#file(GLOB_RECURSE syslib /home/zrji/android_caffe/snap_dragon_lib/sys_lib64/*.so)
#file(GLOB_RECURSE vedorlib /home/zrji/android_caffe/snap_dragon_lib/sys_lib64/vndk-sp/*.so)


message(INFO ${vedorlib})

list(APPEND syslib 
#/home/zrji/android_caffe/snap_dragon_lib/sys_lib64/libc++.so 
#/home/zrji/android_caffe/snap_dragon_lib/sys_lib64/libcutils.so 
#/home/zrji/android_caffe/snap_dragon_lib/sys_lib64/libvndksupport.so
/home/zrji/android_caffe/snap_dragon_lib/sys_lib64/liblog.so
) 



#set(libcutils /home/zrji/android_caffe/snap_dragon_lib/sys_lib64/libcutils.so)
#set(libcpp /home/zrji/android_caffe/snap_dragon_lib/lib64/libc++.so)
#set(libvndksupport /home/zrji/android_caffe/snap_dragon_lib/lib64/libvndksupport.so)

add_definitions(-DUSE_OPENCL)

include(./cmake/Modules/FindOpenCL.cmake)
include_directories(${OPENCL_INCLUDE_DIRS})
list(APPEND Hypertea_LINKER_LIBS ${OPENCL_LIBRARIES} ${syslib}) 




file(GLOB_RECURSE srcs *.cpp)

add_library(hypertea SHARED ${srcs})
target_link_libraries(hypertea ${Hypertea_LINKER_LIBS})
set(Hypertea_LINK hypertea)


#add_executable(main hello_world.cpp)
#target_link_libraries(main ${Hypertea_LINK})
#hypertea_default_properties(main)
