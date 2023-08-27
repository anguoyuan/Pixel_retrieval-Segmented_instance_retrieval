# Content-based pixel retrieval
Pixel retrieval, or segmented instance retrieval, is a new computer vision task; it requires searching pixels that depict the query object from the database. More specifically, it requires the machine to recognize, localize, and segment the query object in database images in run time. It is the pixel-level and instance-level recognition task, as shown in the figure below.
![image](https://github.com/anguoyuan/Pixel_retrieval-Segmented_instance_retrieval/assets/91877920/ae1aa937-0923-4b44-b330-b7010eb28945)
## Compare with image retrieval and segmentation.
Pixel retrieval is an extension of image retrieval and offers information about which pixels are related to the query object. Compared with semantic and instance segmentation, pixel retrieval requires a fine-grained recognition capability for variable-granularity targets. Please take a look at our paper for a more detailed explanation. 
## Try the user-study
Pixel retrieval is a practical task. Our user study results show pixel-level annotation during retrieval can significantly improve the user experience. Try our user-study from [this link](https://fascinating-marzipan-a99b4c.netlify.app/bwds).

# The first two pixel retrieval benchmarks
We introduce the first two pixel retrieval benchmarks named PROxford and PRParis, which are based on the widely used image retrieval datasets, ROxford and RParis. There are several benefits of building new pixel retrieval benchmarks based on the famous ROxford and RParis: 1) Landmarks have great business value; imagine how many figures in your album are about landmarks; 2) It narrows the gap for the image retrieval community to adopt this new task; 3) It is notoriously difficult and each query has up to hundreds positive images, proper to test the instance-level recognition as the benchmark; 4) It has high-quality annotation and can be used to compare human and machine recognition performance. 
## Download dataset
Images and their pixel-level label can be downloaded from this link.

# Pixel retrieval is difficult.
Our extensive experiment results show that the pixel retrieval task is challenging and distinctive from existing problems, suggesting that further research can advance the content-based pixel retrieval and, thus, user search experience. 
## Test baselines
We will open the code and settings to reproduce the result table soon. 

# Possible directions
