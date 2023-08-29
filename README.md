This is the official release of the pixel retrieval benchmarks PRoxford and PRparis, which are introduced in the ICCV 23 paper 'Towards Content-based Pixel Retrieval in Revisited Oxford and Paris'. 

# Access the benchmarks

1. Download our pixel-level labels [here](https://sgvr.kaist.ac.kr/~guoyuan/Segment_retrieval/pixel_retrieval_benchmarks.rar).
2. We provide an example result of 'sift sp' as a demo. Download it from this [link](https://sgvr.kaist.ac.kr/~guoyuan/Segment_retrieval/results.rar).
3. Run miou.py in the evaluation_code directory to get mious for each query.
4. Run mAP@k.py in the evaluation_code directory to get the final mAP@k.
We are still refining this code. 


# Content-based Pixel Retrieval

Pixel retrieval, or segmented instance retrieval, is a new task within computer vision. The primary objective is to identify pixels that represent a specific query object within a database. This involves the machine's ability to recognize, localize, and segment the query object in database images in real-time. Essentially, it's a pixel-level and instance-level recognition task, as illustrated in the figure below:

![Pixel Retrieval Image](https://github.com/anguoyuan/Pixel_retrieval-Segmented_instance_retrieval/assets/91877920/ae1aa937-0923-4b44-b330-b7010eb28945)

## Comparison with Image Retrieval and Segmentation

Pixel retrieval can be viewed as an advanced form of image retrieval, providing insights into which specific pixels correlate with the query object. When juxtaposed with semantic and instance segmentation, pixel retrieval demands a nuanced recognition capability for targets of varying granularity. For a comprehensive understanding, please refer to our [research paper](https://sgvr.kaist.ac.kr/~guoyuan/Segment_retrieval/Towards%20Content-based%20Pixel%20Retrieval%20in%20Revisited%20Oxford%20and%20Paris.pdf).

## User-Study Insights

Pixel retrieval is not just a theoretical concept; it has practical implications. Our user study indicates that pixel-level annotations during the retrieval process can notably enhance user experience. Engage with our user-study through [this link](https://fascinating-marzipan-a99b4c.netlify.app/bwds).

# Introducing Pixel Retrieval Benchmarks

We're excited to present the first two pixel retrieval benchmarks: **PROxford** and **PRParis**. These are built upon the renowned image retrieval datasets, ROxford and RParis. Opting for these datasets offers several advantages:

1. **Relevance to Landmarks**: Landmarks are of immense business value. Consider the number of photos in your collection that feature landmarks.
2. **Familiarity**: Utilizing well-known datasets like ROxford and RParis makes it easier for the image retrieval community to transition to this new task.
3. **Complexity**: The task is inherently challenging. Each query can have hundreds of positive images, making it ideal for testing instance-level recognition.
4. **Quality**: The annotations are of high caliber, facilitating comparisons between human and machine recognition performance.



# Challenges in Pixel Retrieval

Our comprehensive experiments highlight the inherent challenges of the pixel retrieval task. It stands apart from existing challenges, indicating that there's ample scope for research to enhance content-based pixel retrieval and, by extension, the user search experience.

## Upcoming Baselines

Stay tuned! We'll soon release the code and configurations to replicate our results.

# Future Directions

[Our insights or directions for future research or developments.]
