cmake_minimum_required(VERSION 3.12)
enable_language(CUDA)
project(toGreyScale)

# Establecer la ruta de búsqueda de los archivos de encabezado
include_directories(include)

# Agregar los archivos fuente y de encabezado
set(SOURCES src/toGreyScale.cu)
set(BLURRED_SOURCES src/blurred.cu)

set(HEADERS include/utils.hpp)

# Agregar las librerías stb_image y stb_image_write
set(STB_IMAGE include/stb_image.h)
set(STB_IMAGE_WRITE include/stb_image_write.h)

# Compilar el programa
add_executable(toGreyScale ${SOURCES} ${HEADERS} ${STB_IMAGE} ${STB_IMAGE_WRITE})
add_executable(blurred ${BLURRED_SOURCES} ${HEADERS} ${STB_IMAGE} ${STB_IMAGE_WRITE})

# Establecer la configuración de nvcc
set_property(TARGET toGreyScale PROPERTY CUDA_ARCHITECTURES 70)
set_property(TARGET blurred PROPERTY CUDA_ARCHITECTURES 70)

# Agregar una tarea para limpiar
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/CMakeFiles
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/bin
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/lib
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/CMakeCache.txt
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/cmake_install.cmake
    COMMAND ${CMAKE_COMMAND} -E remove ${CMAKE_BINARY_DIR}/Makefile
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)