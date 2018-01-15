
### Project Writeup

[//]: # (Image References)

[image1]: ./pr2_robot/misc images/Normalized confusion matrix.JPG
[image2]: ./pr2_robot/misc images/World 1.JPG
[image3]: ./pr2_robot/misc images/World 2 .JPG
[image4]: ./pr2_robot/misc images/World 3.JPG


### Perception Pipeline
First, a perception.py script was written containing a pcl_callback() function that processes the ROS pointclould msg recieved from the RGB-D camera and a pr2_mover() function to load the simulation scene parameters and request the PR2 robot's PickPlace service that handles the robot's arms movements to pick each identified object and place it in it's correct dropbox.

The pcl_callback() function is called first and it takes a ROS pointcloud msg (from the RGB-D carmera) as input.

First the input is changed from ROS to PCL type.

A statistical outlier filter is then applied to remove some of the artifacts from the data. The filter parameter "number of neighboring points to analyze" was set to 8 and the threshold_scale_factor was set to 0.5.

Voxel grid downsampling was then performed using a voxel size of 0.01 in the x, y and z directions.

Two passthrough filters were then applied. One on the z axis filtering from 0.6m to 1.1m to focus on the tabletop objects in the image and another on the x axis filtering from 0.4 to 1.0 to remove the dropboxes from the field of vision.

RANSAC plane segmentation was then implemented to isolate the objects-cloud(outliers) from the table plane (inlier). A max distance threshold of 0.01 was set.

Euclidean clustering was then performed on the isolated object-clouds (outliers) to further isolate each object into its own cluster. The search method selected was k-d tree. The cluster tolerance, smallest cluster size and Largest Cluster size parameters were set to 0.02, 10 and 800 respectively. Each object cluster was then given a different color for better visualization.

Finally, a SVM machine learning algorithm utilizing histogram features of object colors and surface normals were used to identify each object and give it's correct label.

After machine learning predictions were made the detected objects list was then passed as input to the pr2_mover() function where it was checked for length accuracy and passed, along with several other parameters obtained from the ‘pick_list_*.yaml & dropbox.yaml files, as ROS messages to the PickPlace service. 

Finally, new output_*.yaml files are created containing those same parameters passed to the PickPlace service. 

*  : represents the number 1, 2 or 3 for each simulation scene in this project.

### Feature capture and model training

Histogram features of object colors and surface normal were passed to the SVM algorithm that was used to identify each object.
Both those histogram features were setup under the functions ‘compute_color_histograms()’ and ‘compute_normal_histograms()’  in the ‘features.py’ script. Each function takes a pointcloud as input. The number of histogram bins parameter was set to 32 for both functions and the color type was set to HSV (vs. RBG) for the ‘compute_color_histograms()’ function.
The ‘capture_features.py’ script was used to generate a training set by calling the ‘feature.py’ functions for different poses of each object used in the project. The number of poses was set to 80.
After that, the algorithm was trained using the ‘train_svm.py’ script. The script generates a model.sav file to be used by the ‘perception.py’ script in the project and normalized & non-normalized confusion matrices of the model prediction performance. The best obtained normalized confusion matrix using the mentioned parameters is shown below:

![alt text][image1]

As shown above, the prediction accuracy for the ‘book’, ‘ snacks’ and ‘sticky_notes’ objects have the lowest accuracy.  That was evident when running the project as these objects were sometimes misidentified.
 
### Project outcome
The images below show the pr2_robots attempt to identify the objects in each simulated scene. The number of correct items identified is shown above each image.
Scene 1: 3 of 3 correct idenfitications

![alt text][image2]

Scene 2: 4 of 5 correct identifications

![alt text][image3]

Scene 3: 6 of 8 correct identifications

![alt text][image4]

### Future improvements

Further parameter refining is needed in order to obtain more accurate object identification in the future. One parameter that could be increased in the ‘number of poses’ parameter in the ‘capture_features.py’ script. Generating more training data will most likely help with the model accuracy.

