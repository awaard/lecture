import numpy
import rospy
import random
from std_msgs.msg import Header
from sensor_msgs.msg import JointState
from geometry_msgs.msg import TransformStamped, Transform, Quaternion, Vector3, Point
from visualization_msgs.msg import Marker, InteractiveMarker, InteractiveMarkerControl
from tf import transformations as tf
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from robot_model import RobotModel, Joint
from markers import frame
from linalg import pinv_svd

def alt_orientation_error(T_tgt: numpy.ndarray, T_cur: numpy.ndarray) -> numpy.ndarray:
        """Axis–angle orientation error **ω** in the current end‑effector frame.

        Computes R_err = R_curᵀ · R_tgt and extracts θ·w using
        ``tf.rotation_from_matrix`` as required in Sheet 9.
        """
        R_cur = T_cur[0:3, 0:3]
        R_tgt = T_tgt[0:3, 0:3]
        R_err = R_cur.T @ R_tgt
        T_err = numpy.eye(4)
        T_err[0:3, 0:3] = R_err
        angle, axis, _ = tf.rotation_from_matrix(T_err)
        if numpy.isclose(angle, 0.0):
            return numpy.zeros(3)
        return angle * numpy.array(axis)

class MyInteractiveMarkerServer(InteractiveMarkerServer):
    """Server handling interactive rviz markers"""
    def __init__(self, name, T):
        InteractiveMarkerServer.__init__(self, name)
        self.target = numpy.identity(4)
        self.create_interactive_marker(T)

    def create_interactive_marker(self, T):
        im = InteractiveMarker()
        im.header.frame_id = "world"
        im.name = "target"
        im.description = "Controller Target"
        im.scale = 0.2
        im.pose.position = Point(*T[0:3, 3])
        im.pose.orientation = Quaternion(*tf.quaternion_from_matrix(T))
        self.process_marker_feedback(im)  # set target to initial pose

        # Create a control to move a (sphere) marker around with the mouse
        control = InteractiveMarkerControl()
        control.name = "move_3d"
        control.interaction_mode = InteractiveMarkerControl.MOVE_3D
        control.markers.extend(frame(numpy.identity(4), scale=0.1, frame_id='').markers)
        im.controls.append(control)

        # Create arrow controls to move the marker
        for dir in 'xyz':
            control = InteractiveMarkerControl()
            control.name = "move_" + dir
            control.interaction_mode = InteractiveMarkerControl.MOVE_AXIS
            control.orientation.x = 1 if dir == 'x' else 0
            control.orientation.y = 1 if dir == 'y' else 0
            control.orientation.z = 1 if dir == 'z' else 0
            control.orientation.w = 1
            im.controls.append(control)

            # control = InteractiveMarkerControl()
            # control.name = "rotate_" + dir
            # control.interaction_mode = InteractiveMarkerControl.ROTATE_AXIS
            # control.orientation.x = 1 if dir == "x" else 0
            # control.orientation.y = 1 if dir == "y" else 0
            # control.orientation.z = 1 if dir == "z" else 0
            # control.orientation.w = 1
            # im.controls.append(control)
        # Create a control to rotate the marker


        # Add the marker to the server and indicate that processMarkerFeedback should be called
        self.insert(im, self.process_marker_feedback)

        # Publish all changes
        self.applyChanges()

    def process_marker_feedback(self, feedback):
        """Function called for any marker updates on rviz side"""
        q = feedback.pose.orientation  # marker orientation as quaternion
        p = feedback.pose.position  # marker position
        # Compute target homogenous transform from marker pose
        self.target = tf.quaternion_matrix(numpy.array([q.x, q.y, q.z, q.w]))
        self.target[0:3, 3] = numpy.array([p.x, p.y, p.z])



class Controller(object):
    def __init__(self, pose=TransformStamped(header=Header(frame_id='panda_link8'), child_frame_id='target',
                                             transform=Transform(rotation=Quaternion(*tf.quaternion_about_axis(numpy.pi/4, [0, 0, 1])),
                                                                 translation=Vector3(0, 0, 0.105)))):

        self.robot = RobotModel()
        self.robot._add(Joint(pose))  # add a fixed end-effector transform
        self.pub = rospy.Publisher('/target_joint_states', JointState, queue_size=10)
        self.joint_msg = JointState()
        self.joint_msg.name = [j.name for j in self.robot.active_joints]
        self.joint_msg.position = numpy.asarray(
            [(j.min+j.max)/2 + 0.1*(j.max-j.min)*random.uniform(0, 1) for j in self.robot.active_joints])
        self.target_link = pose.child_frame_id
        self.T, self.J = self.robot.fk(self.target_link, dict(zip(self.joint_msg.name, self.joint_msg.position)))

        self.im_server = MyInteractiveMarkerServer("controller", self.T)

    def actuate(self, q_delta):
        """Move robot by given changes to joint angles"""
        self.joint_msg.position += q_delta.ravel()  # add (numpy) vector q_delta to current joint position vector
        self.pub.publish(self.joint_msg)  # publish new joint state
        joints = dict(zip(self.joint_msg.name, self.joint_msg.position))  # turn list of names and joint values into map
        self.T, self.J = self.robot.fk(self.target_link, joints)  # compute new forward kinematics and Jacobian
    
    
    def solve(self, J, error):
        """Inverse velocity kinematics: q_delta = J^+ * error"""
        return numpy.linalg.pinv(J, 0.01).dot(error)
    
    @staticmethod
    def position_error(T_tgt, T_cur):
        """Translational error (world frame)"""
        return T_tgt[0:3, 3] - T_cur[0:3, 3]

    @staticmethod
    def orientation_error(T_tgt, T_cur):
        """Compute orientation error
        ω = θ · w where θ is the rotation angle and w the rotation axis **expressed in the current
        end‑effector frame**, as required by the exercise sheet (Sheet 9, Task 9.1).
        """
        R_tgt = T_tgt[0:3, 0:3]
        R_cur = T_cur[0:3, 0:3]
        # Rotation bringing current into target, represented in EE frame.
        R_err = R_cur.T.dot(R_tgt)           # R_cur⁻¹ · R_tgt
        # Extend to 4×4 so tf.rotation_from_matrix accepts it
        R_err_h = numpy.eye(4)
        R_err_h[0:3, 0:3] = R_err
        angle, axis, _ = tf.rotation_from_matrix(R_err_h)
        return angle * numpy.asarray(axis)

    # -------------------- Controllers --------------------

    def position_control(self, target):
        v = self.position_error(target, self.T)
        q_delta = self.solve(self.J[0:3, :], v)
        self.actuate(q_delta)

    def pose_control(self, target):
        """6‑D controller combining translation & orientation"""
        v = self.position_error(target, self.T)
        w = self.orientation_error(target, self.T)
        xi = numpy.hstack((v, w))          # 6‑vector twist error
        q_delta = self.solve(self.J, xi)   # full 6×n Jacobian
        self.actuate(q_delta)


if __name__ == "__main__":
    rospy.init_node("ik")
    c = Controller()
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        c.position_control(c.im_server.target)  # full 6‑D control loop
        rate.sleep()
    
