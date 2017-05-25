# generated from genmsg/cmake/pkg-genmsg.cmake.em

message(STATUS "bair_car: 0 messages, 1 services")

set(MSG_I_FLAGS "-Isensor_msgs:/opt/ros/kinetic/share/sensor_msgs/cmake/../msg;-Istd_msgs:/opt/ros/kinetic/share/std_msgs/cmake/../msg;-Igeometry_msgs:/opt/ros/kinetic/share/geometry_msgs/cmake/../msg")

# Find all generators
find_package(gencpp REQUIRED)
find_package(geneus REQUIRED)
find_package(genlisp REQUIRED)
find_package(gennodejs REQUIRED)
find_package(genpy REQUIRED)

add_custom_target(bair_car_generate_messages ALL)

# verify that message/service dependencies have not changed since configure



get_filename_component(_filename "/home/avillaflor/probcoll/robots/rccar/ros/catkin_ws/src/bair_car/srv/sim_env.srv" NAME_WE)
add_custom_target(_bair_car_generate_messages_check_deps_${_filename}
  COMMAND ${CATKIN_ENV} ${PYTHON_EXECUTABLE} ${GENMSG_CHECK_DEPS_SCRIPT} "bair_car" "/home/avillaflor/probcoll/robots/rccar/ros/catkin_ws/src/bair_car/srv/sim_env.srv" "geometry_msgs/Quaternion:sensor_msgs/Image:geometry_msgs/Pose:std_msgs/Header:geometry_msgs/Point"
)

#
#  langs = gencpp;geneus;genlisp;gennodejs;genpy
#

### Section generating for lang: gencpp
### Generating Messages

### Generating Services
_generate_srv_cpp(bair_car
  "/home/avillaflor/probcoll/robots/rccar/ros/catkin_ws/src/bair_car/srv/sim_env.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/bair_car
)

### Generating Module File
_generate_module_cpp(bair_car
  ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/bair_car
  "${ALL_GEN_OUTPUT_FILES_cpp}"
)

add_custom_target(bair_car_generate_messages_cpp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_cpp}
)
add_dependencies(bair_car_generate_messages bair_car_generate_messages_cpp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/avillaflor/probcoll/robots/rccar/ros/catkin_ws/src/bair_car/srv/sim_env.srv" NAME_WE)
add_dependencies(bair_car_generate_messages_cpp _bair_car_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(bair_car_gencpp)
add_dependencies(bair_car_gencpp bair_car_generate_messages_cpp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS bair_car_generate_messages_cpp)

### Section generating for lang: geneus
### Generating Messages

### Generating Services
_generate_srv_eus(bair_car
  "/home/avillaflor/probcoll/robots/rccar/ros/catkin_ws/src/bair_car/srv/sim_env.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/bair_car
)

### Generating Module File
_generate_module_eus(bair_car
  ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/bair_car
  "${ALL_GEN_OUTPUT_FILES_eus}"
)

add_custom_target(bair_car_generate_messages_eus
  DEPENDS ${ALL_GEN_OUTPUT_FILES_eus}
)
add_dependencies(bair_car_generate_messages bair_car_generate_messages_eus)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/avillaflor/probcoll/robots/rccar/ros/catkin_ws/src/bair_car/srv/sim_env.srv" NAME_WE)
add_dependencies(bair_car_generate_messages_eus _bair_car_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(bair_car_geneus)
add_dependencies(bair_car_geneus bair_car_generate_messages_eus)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS bair_car_generate_messages_eus)

### Section generating for lang: genlisp
### Generating Messages

### Generating Services
_generate_srv_lisp(bair_car
  "/home/avillaflor/probcoll/robots/rccar/ros/catkin_ws/src/bair_car/srv/sim_env.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/bair_car
)

### Generating Module File
_generate_module_lisp(bair_car
  ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/bair_car
  "${ALL_GEN_OUTPUT_FILES_lisp}"
)

add_custom_target(bair_car_generate_messages_lisp
  DEPENDS ${ALL_GEN_OUTPUT_FILES_lisp}
)
add_dependencies(bair_car_generate_messages bair_car_generate_messages_lisp)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/avillaflor/probcoll/robots/rccar/ros/catkin_ws/src/bair_car/srv/sim_env.srv" NAME_WE)
add_dependencies(bair_car_generate_messages_lisp _bair_car_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(bair_car_genlisp)
add_dependencies(bair_car_genlisp bair_car_generate_messages_lisp)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS bair_car_generate_messages_lisp)

### Section generating for lang: gennodejs
### Generating Messages

### Generating Services
_generate_srv_nodejs(bair_car
  "/home/avillaflor/probcoll/robots/rccar/ros/catkin_ws/src/bair_car/srv/sim_env.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/bair_car
)

### Generating Module File
_generate_module_nodejs(bair_car
  ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/bair_car
  "${ALL_GEN_OUTPUT_FILES_nodejs}"
)

add_custom_target(bair_car_generate_messages_nodejs
  DEPENDS ${ALL_GEN_OUTPUT_FILES_nodejs}
)
add_dependencies(bair_car_generate_messages bair_car_generate_messages_nodejs)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/avillaflor/probcoll/robots/rccar/ros/catkin_ws/src/bair_car/srv/sim_env.srv" NAME_WE)
add_dependencies(bair_car_generate_messages_nodejs _bair_car_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(bair_car_gennodejs)
add_dependencies(bair_car_gennodejs bair_car_generate_messages_nodejs)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS bair_car_generate_messages_nodejs)

### Section generating for lang: genpy
### Generating Messages

### Generating Services
_generate_srv_py(bair_car
  "/home/avillaflor/probcoll/robots/rccar/ros/catkin_ws/src/bair_car/srv/sim_env.srv"
  "${MSG_I_FLAGS}"
  "/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Quaternion.msg;/opt/ros/kinetic/share/sensor_msgs/cmake/../msg/Image.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Pose.msg;/opt/ros/kinetic/share/std_msgs/cmake/../msg/Header.msg;/opt/ros/kinetic/share/geometry_msgs/cmake/../msg/Point.msg"
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/bair_car
)

### Generating Module File
_generate_module_py(bair_car
  ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/bair_car
  "${ALL_GEN_OUTPUT_FILES_py}"
)

add_custom_target(bair_car_generate_messages_py
  DEPENDS ${ALL_GEN_OUTPUT_FILES_py}
)
add_dependencies(bair_car_generate_messages bair_car_generate_messages_py)

# add dependencies to all check dependencies targets
get_filename_component(_filename "/home/avillaflor/probcoll/robots/rccar/ros/catkin_ws/src/bair_car/srv/sim_env.srv" NAME_WE)
add_dependencies(bair_car_generate_messages_py _bair_car_generate_messages_check_deps_${_filename})

# target for backward compatibility
add_custom_target(bair_car_genpy)
add_dependencies(bair_car_genpy bair_car_generate_messages_py)

# register target for catkin_package(EXPORTED_TARGETS)
list(APPEND ${PROJECT_NAME}_EXPORTED_TARGETS bair_car_generate_messages_py)



if(gencpp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/bair_car)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gencpp_INSTALL_DIR}/bair_car
    DESTINATION ${gencpp_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_cpp)
  add_dependencies(bair_car_generate_messages_cpp sensor_msgs_generate_messages_cpp)
endif()
if(TARGET std_msgs_generate_messages_cpp)
  add_dependencies(bair_car_generate_messages_cpp std_msgs_generate_messages_cpp)
endif()
if(TARGET geometry_msgs_generate_messages_cpp)
  add_dependencies(bair_car_generate_messages_cpp geometry_msgs_generate_messages_cpp)
endif()

if(geneus_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/bair_car)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${geneus_INSTALL_DIR}/bair_car
    DESTINATION ${geneus_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_eus)
  add_dependencies(bair_car_generate_messages_eus sensor_msgs_generate_messages_eus)
endif()
if(TARGET std_msgs_generate_messages_eus)
  add_dependencies(bair_car_generate_messages_eus std_msgs_generate_messages_eus)
endif()
if(TARGET geometry_msgs_generate_messages_eus)
  add_dependencies(bair_car_generate_messages_eus geometry_msgs_generate_messages_eus)
endif()

if(genlisp_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/bair_car)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genlisp_INSTALL_DIR}/bair_car
    DESTINATION ${genlisp_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_lisp)
  add_dependencies(bair_car_generate_messages_lisp sensor_msgs_generate_messages_lisp)
endif()
if(TARGET std_msgs_generate_messages_lisp)
  add_dependencies(bair_car_generate_messages_lisp std_msgs_generate_messages_lisp)
endif()
if(TARGET geometry_msgs_generate_messages_lisp)
  add_dependencies(bair_car_generate_messages_lisp geometry_msgs_generate_messages_lisp)
endif()

if(gennodejs_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/bair_car)
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${gennodejs_INSTALL_DIR}/bair_car
    DESTINATION ${gennodejs_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_nodejs)
  add_dependencies(bair_car_generate_messages_nodejs sensor_msgs_generate_messages_nodejs)
endif()
if(TARGET std_msgs_generate_messages_nodejs)
  add_dependencies(bair_car_generate_messages_nodejs std_msgs_generate_messages_nodejs)
endif()
if(TARGET geometry_msgs_generate_messages_nodejs)
  add_dependencies(bair_car_generate_messages_nodejs geometry_msgs_generate_messages_nodejs)
endif()

if(genpy_INSTALL_DIR AND EXISTS ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/bair_car)
  install(CODE "execute_process(COMMAND \"/home/avillaflor/anaconda2/envs/probcoll/bin/python\" -m compileall \"${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/bair_car\")")
  # install generated code
  install(
    DIRECTORY ${CATKIN_DEVEL_PREFIX}/${genpy_INSTALL_DIR}/bair_car
    DESTINATION ${genpy_INSTALL_DIR}
  )
endif()
if(TARGET sensor_msgs_generate_messages_py)
  add_dependencies(bair_car_generate_messages_py sensor_msgs_generate_messages_py)
endif()
if(TARGET std_msgs_generate_messages_py)
  add_dependencies(bair_car_generate_messages_py std_msgs_generate_messages_py)
endif()
if(TARGET geometry_msgs_generate_messages_py)
  add_dependencies(bair_car_generate_messages_py geometry_msgs_generate_messages_py)
endif()
