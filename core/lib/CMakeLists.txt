file(GLOB SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_library(SemCore ${SOURCES})
add_library(SemCore::SemCore ALIAS SemCore)

target_compile_features(SemCore PUBLIC cxx_std_23)

target_include_directories(
    SemCore
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(SemCore PRIVATE tokenizers_cpp)

if(ONNXRUNTIME_ROOT)
    find_library(
        ONNXRUNTIME_LIBRARY
        NAMES onnxruntime
        PATHS "${ONNXRUNTIME_ROOT}/lib"
        NO_DEFAULT_PATH
        REQUIRED
    )

    add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED GLOBAL)
    set_target_properties(
        ONNXRuntime::ONNXRuntime
        PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_ROOT}/include"
    )

    target_link_libraries(SemCore PRIVATE ONNXRuntime::ONNXRuntime)
endif()
