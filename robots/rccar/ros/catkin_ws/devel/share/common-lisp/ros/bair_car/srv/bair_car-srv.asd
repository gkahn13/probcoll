
(cl:in-package :asdf)

(defsystem "bair_car-srv"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :geometry_msgs-msg
               :sensor_msgs-msg
)
  :components ((:file "_package")
    (:file "sim_env" :depends-on ("_package_sim_env"))
    (:file "_package_sim_env" :depends-on ("_package"))
  ))