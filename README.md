# Vehicle Detection, Tracking and Counting for city traffic analisys 

YOLO v2 std has been exploited for vehicle detection.
Vehicle recognition has been carried out by using sift keypoints and Histogram matching.
Tracking operations are implemetned by means of a TBD(Tracking by detection) approach, still using SIFT keypoints.
Counting and distinction between same class objects (car VS car, truck VS truckl and so on) is executed.
 

If you're using **Pycharm**, after clone the repo use this doc to create the venv in the repository: [https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html), use `./venv/` for the folder name, because it's added on `.gitignore`.

After that, you can easly load the dependencies using: 

Remember: in order to use Tensorflow, Tensorflow-GPU you must use Python 3.6 with Pip at latest version.

Also with pycharm there are some conflict with pip. This can be a solution (after create venv)
```
python -m pip install --upgrade pip

pip install -r venv-configuration.txt
```

If you install other libs, remember to export the new configuration with the following comand: 

```
pip freeze > venv-configuration.txt
```

## Execution
For executing the project, simply

```

python3 flow --model cfg/yolov2.cfg --load models/yolov2.weights --demo videos/test.mp4 --threshold 0.7 --saveVideo --gpu 0.5

```

Where:

- the 1st parameter is the cfg file of the proper yolo version
- the 2nd parameter is the proper trained network (related to the version used as first parameter)
- the 3rd parameter is the input video
- the 4th parameter is the desired threshold, which in this example is set to 0.7
- the 5th parameter is a boolean file which saves the video, eventually
- the 6th parameter regards the GPU use if wanted. The value is related to the amount of GPU and memory to address.

The ouput video is saved in the root of the project and is named 'video.avi.
The files 'times.txt' contains the number of the vehicles detected, tracked and counted in the video and the time interval in which they appear. 


## Note
- Some warning may appear during execution.
- In times.txt file, sometimes the number of the vehicle is doubled: as an example  
```
car11 appears at time: 0.03333333333333333 and disappears at time: 2.933333333333333
```

#Models
Probably it will not be possible to load the models on Github.
In cases you do not find the .h5 file related, please visit the Yolo page.

## Links

- [Configuring Virtualenv Environment Pycharm](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html)
- [Link For Downloading Models](https://pjreddie.com/darknet/yolov2/)



## Author

- Silvio Barra (@silviobarra85)
- Andrea Atzori (@atzoriandrea)



