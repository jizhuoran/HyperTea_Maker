LOCAL_PATH := $(call my-dir)
NDK_PROJECT_PATH := /home/zrji/hypertea_maker/commandline_test
include $(CLEAR_VARS) 
# give module name
LOCAL_MODULE    := hello_world  
# list your C files to compile
LOCAL_SRC_FILES := hello_world.cpp
# this option will build executables instead of building library for android application.
include $(BUILD_EXECUTABLE)