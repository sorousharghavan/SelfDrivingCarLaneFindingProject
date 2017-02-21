#**Finding Lane Lines on the Road** 
###Self-Driving Car Engineer Nanodegree - _Project 1_
###By: **Soroush Arghavan**
---

**Finding Lane Lines on the Road**

The goals of this project are the following:
* Make a pipeline that finds lane lines on the road
* Apply the pipeline to a video feed and output the resulting video with the lane overlay
* Analyze the shortcomings and improvement strategies


[//]: # (Image References)

[image1]: ./figure_0.png "Grayscale"
[image2]: ./figure_1.png "Canny Edge Detection"
[image3]: ./figure_2.png "Masked image"
[image4]: ./figure_3.png "Masking each half to focus on one lane at a time"
[image5]: ./figure_4.png "Finding lines using Hough transform"
[image6]: ./figure_5.png "Extrapolated lanes"
[image7]: ./figure_6.png "Final output"

---

### Reflection

###1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 5 steps. First, the input image has to be converted into a grayscale matrix. This is necessary in order to simplify and speed up the processing in the later stages.

![image1]

Second step of the pipeline is applying Gaussian blurring in order to reduce noise. This step depends on the quality of the input image and the road conditions. During experimentation with different resolutions up to 2K, it was found that there is no significant correlation between the resolution of the input image
and the suitable kernel size. Lower kernel sizes appeared to be working as well as higher kernel sizes on high resolution input.

![image2]

Furthermore, a mask is applied on the frame in order to focus on the region of interest. A trapezoidal mask is preferred over a triangular mask since it matches the shape of the lanes in the frames better.

![image3]

In order to draw a single line on the left and right lanes, I modified the draw_lines() function by processing the data further through **three** additional functions.
First, the _y_ coordinate of the line segments are compared against the midpoint of the frame to determine whether the point/line belongs to the lane to the right of the vehicle or the left. This allows us to be able to distinguish between the two lanes.

![image4]

Secondly, the data is passed to an averaging function, where the lanes are averaged over the last few frames.
The higher the averaging frames, the more stable the displayed lane. However, increasing the number of averaged frames increaases lag in lane movements significantly. This poses an issue in circumstances where lanes change rapidly from frame to frame such as
turns. Secondly, the lane segments are extrapolated to cover the whole lane with one solid line. In order to achieve this, the top-most (lowest _x_ in Cartesian coordinates) and the bottom-most (highest _x_) are found, the connecting line's slope and intercept
are calculated and the line is extrapolated to the bottom of the frame (where `x = height`). The resulting lanes are shown below.

![image5]

![image6]

Finally, the lanes are overlayed on the original frame and returned by the pipeline.

![image7]

###2. Testing on my own footage

I used the footage from my own dashcam in order to test the pipeline. Although the pipeline had to be recalibrated and the output was not as stable as one would hope, the pipeline proved some levels of success. The video output is available below.

[Test 1](https://youtu.be/0zsHVEKrzDE)
[Test 2](https://youtu.be/tSJuFduDjcc)

In test 1, it can be seen that the shadow from the passing car causes inaccuracy in data. However, the effect is not very severe and is damped by averaging the data to some extent. In test 2, the effect of a turn on detection pipeline can be seen. Since the lane curvature is too high and the lane is outside of the masking region, we can see that the right lane detection becomes inaccurate.

###3. Identify potential shortcomings with your current pipeline

After testing the pipeline on several video feeds from the examples as well as my own footage, the following shortcomings were noticed:
* The pipeline has to be recalibrated for different cameras, mounting positions and lighting conditions. Parameters such as the kernel size for Gaussian blur, Hough transform parameters and the Canny edge detection are sensitive to such conditions.
* The pipeline will not work in direct sunlight. A physical mask is the minimum requirement to prevent block sunlight.
* During the first iteration of the pipeline, I decided to brute-force scan the image to find the two ends of lanes to extrapolate. This process was extremely time consuming and reduced to iteration time to 5 seconds per iteration. By improving the scanning, this number was dropped to 2 seconds per iteration which is still not acceptable. The current method has a ~34 frame/second rate which is fairly okay but not high enough for real-time applications. Considering that more processing has to be done on each frame to detect objects, run data through classifiers and PID controllers, this frame rate would not be sufficient to perform a real-time analysis.
* The masking region is static in the current method. The lanes would not be detected correctly if the car moves off-center or when the vehicle is in the process of changing lanes.
* If another vehicle moves in front of the vehicle within the region of interest, the lane data will no longer be accurate.

###4. Suggest possible improvements to your pipeline

* To increase the framerate, using a low level language such as C might be more efficient. Using a lower input resolution would also be effective.
* A more complex masking method could improve the detection under unsteady lighting conditions. One method that is worth exploring is to use the previous frame to _guess_ the position of the lanes in the current frame where the current frame might be noisy.
* Using alternative spectrums of light other than visible light might also be an option in locations where the lanes are reflective. Using infrared or UV cameras could be more accurate since they could be less prone to road surface conditions.
* Applying a low-pass filter to the lanes coordinates could also improve the stability, since the lane changes are often gradual and not sudden.
