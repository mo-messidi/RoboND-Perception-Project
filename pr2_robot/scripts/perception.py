#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

    cloud = ros_to_pcl(pcl_msg) #Convert ROS msg to PCL data


    # Statistical Outlier Filtering
    outlier_filter = cloud.make_statistical_outlier_filter()
    neighboring_pts_to_analyze = 8
    threshold_scale_factor = 0.5
    outlier_filter.set_mean_k(neighboring_pts_to_analyze)
    outlier_filter.set_std_dev_mul_thresh(threshold_scale_factor)

    cloud_filtered = outlier_filter.filter() # resulting filtered point cloud
    

    # Voxel Grid Downsampling
    vox = cloud_filtered.make_voxel_grid_filter()

    LEAF_SIZE = 0.01 # voxel size 
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)

    cloud_filtered = vox.filter() #resulting downsampled point cloud


    # PassThrough Filter
    passthrough = cloud_filtered.make_passthrough_filter()

    # z-axis passthrough filter for table elimination
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)  

    cloud_filtered = passthrough.filter() # resulting filtered point cloud

    # x-axis passthrough filter for dropboxes elimination
    filter_axis = 'x'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.4
    axis_max = 1.0
    passthrough.set_filter_limits(axis_min, axis_max)

    cloud_filtered = passthrough.filter() # resulting filtered point cloud

    
    # RANSAC Plane Segmentation
    seg = cloud_filtered.make_segmenter()
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)

    max_distance = 0.01
    seg.set_distance_threshold(max_distance)

    inliers, coefficients = seg.segment() # set of inlier indices and model coefficients

    cloud_table = cloud_filtered.extract(inliers, negative=False) # Extract inliers
    ros_cloud_table = pcl_to_ros(cloud_table) # convert PCL data to ROS messages

    cloud_objects = cloud_filtered.extract(inliers, negative=True) # Extract outliers
    ros_cloud_objects = pcl_to_ros(cloud_objects) # convert PCL data to ROS messages


    # Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
    ec = white_cloud.make_EuclideanClusterExtraction()

    ec.set_ClusterTolerance(0.02)
    ec.set_MinClusterSize(10)
    ec.set_MaxClusterSize(800)
    ec.set_SearchMethod(tree)

    cluster_indices = ec.Extract() # extract indices for each discovered cluster


    # Visualize Clusters
    cluster_color = get_color_list(len(cluster_indices)) # assign a color for each segmented object

    color_cluster_point_list = []
    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])

    cluster_cloud = pcl.PointCloud_PointXYZRGB() # create new cloud object
    cluster_cloud.from_list(color_cluster_point_list) # assign containing all clusters, each with unique color
    ros_cloud_cluster = pcl_to_ros(cluster_cloud) # convert PCL data to ROS messages


    # Classify Clusters
    detected_objects_labels = []
    detected_objects_list = []

    for index, pts_list in enumerate(cluster_indices):
        pcl_cluster = cloud_objects.extract(pts_list)
        ros_cluster = pcl_to_ros(pcl_cluster) # convert PCL data to ROS messages

        # Extract histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=True)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))

        prediction = clf.predict(scaler.transform(feature.reshape(1,-1))) # Make the prediction using support vector machines model
        label = encoder.inverse_transform(prediction)[0] # get labels from predictions
        detected_objects_labels.append(label) #append labels list

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects_list.append(do)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

   
    # Publish ROS messages
    detected_objects_pub.publish(detected_objects_list)
    pcl_objects_pub.publish(ros_cloud_objects)
    pcl_table_pub.publish(ros_cloud_table)
    pcl_cluster_pub.publish(ros_cloud_cluster)

    try:
        pr2_mover(detected_objects_list)
    except rospy.ROSInterruptException:
        pass



# Function to load parameters and request PickPlace service
def pr2_mover(object_list):

    # Get ROS parameters 
    object_list_param = rospy.get_param('/object_list')
    scene_num = 1 # CHANGE FOR EACH SCENE (1, 2, 3) LOADED
    dropbox = rospy.get_param('/dropbox')


    # Check that dected objects match actual objects list
    if not len(object_list) == len(object_list_param):
        rospy.loginfo("List of detected objects does not match pick list.")
        return 

    # Initialize variables
    pick_pose = Pose()
    place_pose = Pose()
    object_name = String()
    arm_name = String()
    test_scene_num = Int32()

    # Parse parameter into individual variable
    test_scene_num.data = scene_num

    centroids = [] 
    params_out = [] # list of parameters to be outputed to yaml file
    i = 0  # counter variable
    for object in object_list:

    	# Get the PointCloud for a given object and obtain it's centroid
    	points_arr = ros_to_pcl(object.cloud).to_array()
    	centroids.append((np.mean(points_arr, axis=0))[:3])
    	
        # Convert numpy array to python native type
        centroid = centroids[i]
        float_centroid = [np.asscalar(j) for j in centroid]

        pick_pose.position.x = float_centroid[0]
        pick_pose.position.y = float_centroid[1]
        pick_pose.position.z = float_centroid[2]

        # Parse 'group' parameter into individual variable
    	object_group = str(object_list_param[i]['group'])

    	# Parse 'name' parameter into individual variable
    	object_name.data = str(object_list_param[i]['name'])

    	# Parse 'arm selector' parameter into individual variable
    	arm_name.data = str('right' if object_group == 'green' else 'left')

        # Parse 'dropbox pos' parameter into individual variable
        dropbox_position = dropbox[0]['position'] if object_group == 'red' else dropbox[1]['position']
        place_pose.position.x = dropbox_position[0]
        place_pose.position.y = dropbox_position[1]
        place_pose.position.z = dropbox_position[2] 

        # Create a list of dictionaries (made with make_yaml_dict()) for later output to yaml format
        params_out.append(make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose))

        # Wait for 'pick_place_routine' service to come up
        rospy.wait_for_service('pick_place_routine')

        try:
            pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)
            # Insert message variables to be sent as a service request
            resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)
            print ("Response: ",resp.success)
        except rospy.ServiceException, e:
            print "Service call failed: %s"%e

        i += 1 #iterate counter variable

    # Output request parameters into output yaml file
    file_name = "output_{}.yaml".format(scene_num)
    send_to_yaml(file_name, params_out)


if __name__ == '__main__':

    # ROS node initialization
    rospy.init_node('object_recognition', anonymous=True)

    # Create subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)

    # Create publishers
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=1)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)

    # Load model from disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler']

    # Initialize color_list
    get_color_list.color_list = []

    # Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()
