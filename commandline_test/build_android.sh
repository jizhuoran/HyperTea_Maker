export NDK_HOME="/home/zrji/android_caffe/tmp_ndk/android-ndk-r18b"
export DEVICE_OPENCL_DIR="/home/zrji/android_caffe/caffe-android-opencl/third_party/OpenCL/"



export ANDROID_ABI=arm64-v8a
export ANDROID_NATIVE_API_LEVEL=26

mkdir ./build_${ANDROID_ABI%% *}
cd ./build_${ANDROID_ABI%% *} || exit 1
rm -rf *
cmake .. -DCMAKE_TOOLCHAIN_FILE=$NDK_HOME/build/cmake/android.toolchain.cmake \
    -DANDROID_NDK=$NDK_HOME \
    -DANDROID_ABI="$ANDROID_ABI" \
    -DANDROID_NATIVE_API_LEVEL=$ANDROID_NATIVE_API_LEVEL \
    -DOPENCL_ROOT=$DEVICE_OPENCL_DIR \
    -G "Unix Makefiles" || exit 1
make -j 40 || exit 1