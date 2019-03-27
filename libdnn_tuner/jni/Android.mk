LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS) 
LOCAL_MODULE := libhypertea
LOCAL_EXPORT_C_INCLUDES := /home/zrji/hypertea_maker/libdnn_tuner/sources
LOCAL_SRC_FILES := /home/zrji/hypertea_maker/libdnn_tuner/sources/build_arm64-v8a/libhypertea.so
include $(PREBUILT_SHARED_LIBRARY)


include $(CLEAR_VARS) 
LOCAL_MODULE := opencl
LOCAL_EXPORT_C_INCLUDES := /home/zrji/android_caffe/caffe-android-opencl/third_party/OpenCL/include
LOCAL_SRC_FILES := /home/zrji/android_caffe/caffe-android-opencl/third_party/OpenCL/lib64/libOpenCL.so
include $(PREBUILT_SHARED_LIBRARY)


include $(CLEAR_VARS) 
LOCAL_MODULE := cutils
LOCAL_SRC_FILES := /home/zrji/android_caffe/snap_dragon_lib/sys_lib64/libcutils.so
include $(PREBUILT_SHARED_LIBRARY)


include $(CLEAR_VARS) 
LOCAL_MODULE := libvndksupport
LOCAL_SRC_FILES := /home/zrji/android_caffe/snap_dragon_lib/sys_lib64/libvndksupport.so
include $(PREBUILT_SHARED_LIBRARY)






NDK_PROJECT_PATH := /home/zrji/hypertea_maker/libdnn_tuner/jni/
# give module name
LOCAL_MODULE    := main  
# list your C files to compile
LOCAL_SRC_FILES := main.cpp

LOCAL_SHARED_LIBRARIES := libhypertea opencl cutils libvndksupport
# this option will build executables instead of building library for android application.
include $(BUILD_EXECUTABLE)
#