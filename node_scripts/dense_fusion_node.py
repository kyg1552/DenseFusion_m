#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import cameramodels
import cv2
import cv_bridge
import jsk_recognition_msgs.msg
import message_filters
import numpy as np
import rospy
import sensor_msgs.msg
import visualization_msgs.msg

from dense_fusion.datasets.ycb.ycb_utils import get_object_pcds
from dense_fusion.models import DenseFusion
from dense_fusion.visualizations.vis_bboxes import voc_colormap
from dense_fusion.visualizations import vis_pcd #visualization point could data

class DenseFusionNode(object):
    

    def __init__(self):
        super(DenseFusionNode, self).__init__()

        self.num_obj = 21
        self.model = DenseFusion()
        self.model.cuda()
        self.model.eval()

        self.target_label = rospy.get_param('~target_label', '025_mug')
        self.mesh_model_path = rospy.get_param('~mesh_model_path', None)
        print(self.mesh_model_path)
        if self.mesh_model_path is None:
            self.markers_pub = None
        else:
            self.markers_pub = rospy.Publisher(
                '~output/markers',
                visualization_msgs.msg.MarkerArray,
                queue_size=1)
        self.image_pub = rospy.Publisher('~output/viz',
                                         sensor_msgs.msg.Image,
                                         queue_size=1)
        
        self.bridge = cv_bridge.CvBridge()
        self.cameramodel = None
        self.color_map = voc_colormap(self.num_obj + 1, order='bgr')
        self.object_pcds = get_object_pcds()
        
        self.subscribe()
        
    def subscribe(self):
        
        queue_size = rospy.get_param('~queue_size', 5)
        sub_img = message_filters.Subscriber(
            '~input/image',
            sensor_msgs.msg.Image,
            queue_size=1,
            buff_size=2**24)
        sub_label_img = message_filters.Subscriber(
            '~input/label',
            sensor_msgs.msg.Image,
            queue_size=1,
            buff_size=2**24)
        sub_depth = message_filters.Subscriber(
            '~input/depth',
            sensor_msgs.msg.Image,
            queue_size=1,
            buff_size=2**24)
        sub_rects = message_filters.Subscriber(
            '~input/rects',
            jsk_recognition_msgs.msg.RectArray,
            queue_size=1,
            buff_size=2**24)
        sub_class = message_filters.Subscriber(
            '~input/class',
            jsk_recognition_msgs.msg.ClassificationResult,
            queue_size=1,
            buff_size=2**24)
        self.subs = [sub_img, sub_label_img, sub_depth, sub_rects, sub_class]

        if rospy.get_param('~approximate_sync', False):
            slop = rospy.get_param('~slop', 0.1)
            sync = message_filters.ApproximateTimeSynchronizer(
                fs=self.subs, queue_size=queue_size, slop=slop)
        else:
            sync = message_filters.TimeSynchronizer(
                fs=self.subs, queue_size=queue_size)
        sync.registerCallback(self._cb)

    # def unsubscribe(self):
    #     for s in self.subs:
    #         s.unregister()


    def _cb(self, img_msg, label_msg, depth_msg, rects_msg, class_msg):
        if self.cameramodel is None:
            rospy.loginfo("Waiting camera info ...")
            return
        bridge = self.bridge
        try:
            rgb_img = bridge.imgmsg_to_cv2(img_msg, 'rgb8')
            depth_img = bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            label_img = bridge.imgmsg_to_cv2(label_msg)

            if depth_msg.encoding == '16UC1':
                depth_img = np.asarray(depth_img, dtype=np.float32)
                depth_img /= 1000.0  # convert metric: mm -> m
            elif depth_msg.encoding != '32FC1':
                rospy.logerr('Unsupported depth encoding: %s' %
                             depth_msg.encoding)
        except cv_bridge.CvBridgeError as e:
            rospy.logerr('{}'.format(e))
            return

        bboxes = []
        labels = []
        for rect_msg, label in zip(rects_msg.rects,
                                   class_msg.label_names):
            if label == self.target_label:
                y_min = rect_msg.y
                x_min = rect_msg.x
                y_max = y_min + rect_msg.height
                x_max = x_min + rect_msg.width
                bboxes.append([y_min, x_min, y_max, x_max])
                labels.append(
                    class_msg.target_names.index(label))
        bboxes = np.array(bboxes)
        labels = np.array(labels, dtype=np.int32)
        bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

        intrinsic_matrix = self.cameramodel.K
        # print("camera info{}".foramt(intrinsic_matrix))
        rotations, translations = self.model.predict(
            rgb_img, depth_img, label_img,
            labels, bboxes, intrinsic_matrix)

        print("rot:{0}, trans: {1}".format(rotation, translation))

if __name__ == '__main__':
    rospy.init_node('dense_fusion_node')
    DenseFusionNode()
    rospy.spin()
