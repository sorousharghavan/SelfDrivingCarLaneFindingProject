from helpers import *

dirPath = "test_images/"
files = os.listdir(dirPath)

for file in files:
    image = mpimg.imread(dirPath + file)
    #image = process_image(image)
    #show_image(image)

white_output = 'white.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)

#yellow_output = 'yellow.mp4'
#clip2 = VideoFileClip('solidYellowLeft.mp4')
#yellow_clip = clip2.fl_image(process_image)
#yellow_clip.write_videofile(yellow_output, audio=False)

#challenge_output = 'chal.mp4'
#clip2 = VideoFileClip('challenge.mp4')
#challenge_clip = clip2.fl_image(process_image)
#challenge_clip.write_videofile(challenge_output, audio=False)
    
#challenge_output = 'jetta2.mp4'
#clip2 = VideoFileClip('test_images_jetta/VICO2984.mp4')
#challenge_clip = clip2.fl_image(process_image)
#challenge_clip.write_videofile(challenge_output, audio=False)    
