<div align="center">
<img src="https://github.com/anguoyuan/Pixel_retrieval-Segmented_instance_retrieval/assets/91877920/6c9cc2f4-58e1-4fe6-a97c-bfb173479f16" width="200" height="100" align="middle">
</div>


# Official Release: Pixel Retrieval Benchmarks PRoxford and PRparis 

We are pleased to announce the official release of the pixel retrieval benchmarks PRoxford and PRparis. These benchmarks are introduced in our ICCV 23 paper, "Towards Content-based Pixel Retrieval in Revisited Oxford and Paris."

## Participate in Our User Study

**Challenge**: Can you identify the left query object in the right ground truth image?

Traditional image-level retrieval often falls short in meeting user search requirements. For a more precise and user-centric experience, pixel-level retrieval is essential.

![Query Image](https://github.com/anguoyuan/Pixel_retrieval-Segmented_instance_retrieval/assets/91877920/dfc800bf-5548-453d-a97c-0cfa373cdad5) ![Ground Truth Image](https://github.com/anguoyuan/Pixel_retrieval-Segmented_instance_retrieval/assets/91877920/ed325955-a58f-4fa1-8bd2-78bedb0abcff)

Pixel retrieval isn't merely theoretical—it has tangible benefits. Our research shows that incorporating pixel-level annotations during the retrieval process significantly improves user experience.

**Get Involved**: Dive into our user study and share your insights [here](https://fascinating-marzipan-a99b4c.netlify.app/bwds).



## Accessing the Benchmarks

### Prerequisites

1. **Download Pixel-Level Labels**: Download and extract our pixel-level labels from [here](https://sgvr.kaist.ac.kr/~guoyuan/Segment_retrieval/pixel_retrieval_benchmarks.rar).
   
2. **Download Example Result**: We provide an example result using the 'sift sp' method. Download and extract it from [this link](https://sgvr.kaist.ac.kr/~guoyuan/Segment_retrieval/results.rar).

### Evaluation Steps

1. Run `miou.py` in the `evaluation_code` directory to get mIoUs for each query.
   
2. Run `mAP@k.py` in the `evaluation_code` directory to get the final mAP@k.

### Testing Your Own Method

To test your own method, named 'ABC':
- Save the image-level and pixel-level results in `results/image-level/ABC` and `results/pixel-level/ABC/matching_result`, respectively.
- Modify the method name in `miou.py` and `mAP@k.py`. Run them.

### Directory Structure

Your directory structure should be as follows:

-- pixel_retrieval-Segmented_instance_retrieval

|-- pixel_retrieval_benchmarks

|-- evaluation_code

|-- results

> **Note**: We are still refining this evaluation code. If you encounter any issues while using it, please let us know. We also plan to release configurations and code to help readers replicate the results presented in our paper. Stay tuned for updates.





# Content-based Pixel Retrieval

Pixel retrieval, or segmented instance retrieval, is a new task within computer vision. The primary objective is to identify pixels that represent a specific query object within a database. This involves the machine's ability to recognize, localize, and segment the query object in database images in real-time. Essentially, it's a pixel-level and instance-level recognition task, as illustrated in the figure below:

![Pixel Retrieval Image](https://github.com/anguoyuan/Pixel_retrieval-Segmented_instance_retrieval/assets/91877920/ae1aa937-0923-4b44-b330-b7010eb28945)

## Comparison with Image Retrieval and Segmentation

Pixel retrieval can be viewed as an advanced form of image retrieval, providing insights into which specific pixels correlate with the query object. When juxtaposed with semantic and instance segmentation, pixel retrieval demands a nuanced recognition capability for targets of varying granularity. For a comprehensive understanding, please refer to our [research paper](https://sgvr.kaist.ac.kr/~guoyuan/Segment_retrieval/Towards%20Content-based%20Pixel%20Retrieval%20in%20Revisited%20Oxford%20and%20Paris.pdf).


# Introducing Pixel Retrieval Benchmarks

We're excited to present the first two pixel retrieval benchmarks: **PROxford** and **PRParis**. These are built upon the renowned image retrieval datasets, ROxford and RParis. Opting for these datasets offers several advantages:

1. **Relevance to Landmarks**: Landmarks are of immense business value. Consider the number of photos in your collection that feature landmarks.
2. **Familiarity**: Utilizing well-known datasets like ROxford and RParis makes it easier for the image retrieval community to transition to this new task.
3. **Complexity**: The task is inherently challenging. Each query can have hundreds of positive images, making it ideal for testing instance-level recognition.
4. **Quality**: The annotations are of high caliber, facilitating comparisons between human and machine recognition performance.



# Challenges in Pixel Retrieval

Our comprehensive experiments highlight the inherent challenges of the pixel retrieval task. It stands apart from existing challenges, indicating that there's ample scope for research to enhance content-based pixel retrieval and, by extension, the user search experience.

# Future Directions

[Our insights or directions for future research or developments.]
