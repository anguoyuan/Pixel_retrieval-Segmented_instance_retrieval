<div align="center">
<img src="https://github.com/anguoyuan/Pixel_retrieval-Segmented_instance_retrieval/assets/91877920/6c9cc2f4-58e1-4fe6-a97c-bfb173479f16" width="200" height="100" align="middle">
</div>


# Official Release: Pixel Retrieval Benchmarks PRoxford and PRparis 

We are pleased to announce the official release of the pixel retrieval benchmarks PRoxford and PRparis. These benchmarks are introduced in our ICCV 23 paper, "Towards Content-based Pixel Retrieval in Revisited Oxford and Paris."

## Participate in Our User Study

**Challenge**: Can you identify the left query object in the right ground truth image?

Traditional image-level retrieval often falls short in meeting user search requirements. For a more precise and user-centric experience, pixel-level retrieval is essential.

<img src="https://github.com/anguoyuan/Pixel_retrieval-Segmented_instance_retrieval/assets/91877920/dfc800bf-5548-453d-a97c-0cfa373cdad5" alt="Query Image" width="40%" align="left"> <img src="https://github.com/anguoyuan/Pixel_retrieval-Segmented_instance_retrieval/assets/91877920/ed325955-a58f-4fa1-8bd2-78bedb0abcff" alt="Ground Truth Image" width="45%" align="right">

<br clear="all">

Pixel retrieval isn't merely theoretical—it has tangible benefits. Our research shows that incorporating pixel-level annotations during the retrieval process significantly improves user experience.

**Get Involved**: Dive into our user study and share your insights [here](https://fascinating-marzipan-a99b4c.netlify.app/bwds).



## Accessing the Benchmarks

### Prerequisites

1. **Download Pixel-Level Labels**: Download and unzip our pixel-level labels from [here](https://sgvr.kaist.ac.kr/~guoyuan/Segment_retrieval/pixel_retrieval_benchmarks.rar).
   
2. **Download Example Result**: We provide an example result using the 'sift sp' method. Download and extract it from [this link](https://sgvr.kaist.ac.kr/~guoyuan/Segment_retrieval/results.rar).

3. **Revisited Oxford and Paris datasets**: As the official link of oxford and paris is not availabel now, we provide an alternative [download link](https://sgvr.kaist.ac.kr/~guoyuan/Segment_retrieval/revisitop.zip).

### Evaluation Steps

1. Run `miou.py` in the `evaluation_code` directory to get mIoUs for each query.
   
2. Run `mAP@k.py` in the `evaluation_code` directory to get the final mAP@k.

### Testing Your Own Method

To test your own method, named 'ABC':
- Save the image-level and pixel-level results in `results/image-level/ABC` and `results/pixel-level/ABC/matching_result`, respectively.
- The pixel-level result is the ranking array with shape (#db_imgs, #query_imgs).
- The pixel-level result is a dictionary saved using numpy, e.g., "np.save('proxford.npy', dictionary)". The dictionary key is like 'qimlist_69_imlist_1534', which means the matching result of query 69 and db 1534. Dictionary['qimlist_69_imlist_1534'] is a numpy array with shape (#boundary_keypoints, 2), recording the location (width and height) of the boundary keypoints of the segmented instance in the db_img 1534. The boundary can be either a box or a polygon. The keypoints can also contain the matching points inside the instance.   
- Modify the method name in `miou.py` and `mAP@k.py`. Run them.

### Directory Structure

Your directory structure should be as follows:

-- pixel_retrieval-Segmented_instance_retrieval

|-- pixel_retrieval_benchmarks

|-- evaluation_code

|-- results

> **Note**: We are still refining this evaluation code. If you encounter any issues while using it, please let us know. We also plan to release configurations and code to help readers replicate the results presented in our paper. Stay tuned for updates.


### updated testbed
<details>
<summary>Click to expand LaTeX code</summary>
\begin{table*}[htp]
    \caption{Results of pixel retrieval from ground truth query-index image pairs (\% mean of mIoU) on the PROxf/PRPar datasets with both Medium and Hard evaluation protocols. D and S indicate detection and segmentation results respectively. \textbf{Bold} number indicates the best performance, and \underline{underline} indicates the second one.  
    }
    \label{tab:miou}
    \centering
    \newcolumntype{C}[1]{>{\centering\arraybackslash}p{#1}}
    \begin{tabular}{|l | c | c | c | c | c | c | c | c| c|}
    %\begin{tabular*}{\textwidth}{|C{0.354\textwidth}|C{0.045\textwidth}|C{0.045\textwidth}|C{0.045\textwidth}|C{0.045\textwidth}|C{0.045\textwidth}|C{0.045\textwidth}|C{0.045\textwidth}|C{0.045\textwidth}|C{0.045\textwidth}|}
    \hline
   \multirow{3}{*}{Method} & \multicolumn{4}{c}{Medium} & \multicolumn{4}{|c|}{Hard}& \\
    \cline{2-10}
     & \multicolumn{2}{c|}{PROxf}  & \multicolumn{2}{c|}{PRPar} & \multicolumn{2}{c|}{PROxf} & \multicolumn{2}{c|}{PRPar} &Average \\
     \cline{2-10}
          & D & S & D & S & D & S & D & S &\\
    \hline
    \multicolumn{10}{|c|}{ Retrieval and localization unified methods}\\
    \hline
        SIFT+SP~\cite{philbin2007object} &26.1  &10.9 &24.2 &9.7 &18.2  &7.3  &19.3 &7.8&15.44 \\
        DELF+SP~\cite{Noh_2017_ICCV} & \underline{43.7} &20.0  &\textbf{40.7} &16.7 &\underline{33.2} &13.9 &32.2 &12.4&26.60 \\
        DELG+SP~\cite{cao2020unifying} &\textbf{44.1} &19.7  &\underline{40.1} &16.5 &\textbf{34.8} &14.5 &31.2 &11.7& 26.57 \\
        D2R~\cite{teichmann2019detect}+Resnet-50-Faster-RCNN+Mean  &20.2 &-  &29.6 &- &16.7 &-  &27.4 &-&- \\
        D2R~\cite{teichmann2019detect}+Resnet-50-Faster-RCNN+VLAD~\cite{jegou2010aggregating} &25.8  &- &37.5 &- &21.6  &-  &\underline{35.5} &-&- \\
        D2R~\cite{teichmann2019detect}+Resnet-50-Faster-RCNN+ASMK~\cite{tolias2016image} &26.3  &-  &38.5 &- &21.6 &- &\textbf{35.6} &-&-\\
        D2R~\cite{teichmann2019detect}+Mobilenet-V2-SSD+Mean &19.7 &-  &25.9 & -&20.1 &-  &27.9 &-&- \\
        D2R~\cite{teichmann2019detect}+Mobilenet-V2-SSD+VLAD~\cite{jegou2010aggregating}  &23.1 &-  &33. &- &20.9 &-  &33.6 &-&- \\
        D2R~\cite{teichmann2019detect}+Mobilenet-V2-SSD+ASMK~\cite{tolias2016image}   &22.4 & - &34.0 &- &20.8 &-  &33.1 &-&- \\
    \hline
    \multicolumn{10}{|c|}{Detection methods}\\
    \hline
    OWL-VIT (LiT)~\cite{minderer2022simple} & 11.4 & - & 18.0 & - & 6.3 & - & 15.0 & -&- \\
    OS2D-v2-trained~\cite{osokin20os2d} & 10.5 &-  &13.7 &- &11.7 &- &14.3 &-&-\\
    OS2D-v1~\cite{osokin20os2d}  &7.0 &-  &8.5 &- & 8.7&-  &9.2 &-&- \\
    OS2D-v2-init~\cite{osokin20os2d}   &13.6 &-  &15.4 &- &14.0 &-  &15.1 & -&-\\
\hline
    \multicolumn{10}{|c|}{Segmentation methods}\\
    \hline
    SSP (COCO) + ResNet50~\cite{fan2022ssp} & 19.2 & \underline{34.5} & 31.1 & \underline{48.7} & 15.1 & \underline{25.3} & 29.8 & \textbf{41.7}&30.68 \\
     SSP (VOC) + ResNet50~\cite{fan2022ssp}  & 19.7 & 34.3 & 31.4 & \textbf{48.8} & 16.1 & \textbf{26.1} & 30.3 & \underline{40.4}&30.89 \\
     HSNet (COCO) + ResNet50~\cite{min2021hypercorrelation} & 23.4 & 32.8 & 37.4 & 41.9 & 21.0 & 25.7 & 34.7 & 36.5&\underline{31.67} \\
     HSNet (VOC) + ResNet50~\cite{min2021hypercorrelation} & 21.0 & 29.8 & 31.4 & 39.7 & 17.1 & 23.2 & 29.7 & 34.9&28.35 \\
     HSNet (FSS) + ResNet50~\cite{min2021hypercorrelation} & 30.5 & \textbf{35.7} & 39.4 & 40.2 & 22.7 & 25.1 & 34.7 & 32.8&\textbf{32.64} \\
     Mining (VOC) + ResNet50~\cite{yang2021mining} & 18.3 & 30.5 & 29.6 & 42.7 & 15.1 & 21.4 & 28.1 & 34.3&27.50 \\
     Mining (VOC) + ResNet101~\cite{yang2021mining} & 18.1 & 28.6 & 29.5 & 40.0 & 14.2 & 20.4 & 28.2 & 34.4&26.68 \\
     % m8  & &  & & & &  & & \\
        \hline
    \multicolumn{10}{|c|}{Dense matching methods}\\
    \hline
    GLUNet-Geometric~\cite{truong2020glu} & 18.1 & 13.2 & 22.8 & 15.2 & 7.7 & 4.6 & 13.3 & 7.8&12.84 \\
    PDCNet-Geometric~\cite{truong2021pdc} & 29.1 & 24.0 & 30.7 & 21.9 & 20.4 & 15.7 & 20.6 & 12.6&21.87 \\
    GOCor-GLUNet-Geometric~\cite{truong2020gocor} & 30.4 & 26.0 & 33.4 & 25.6 & 20.8 & 16.0 & 19.8 & 13.3&23.16 \\
    WarpC-GLUNet-Geometric (megadepth)~\cite{truong2021warp} & 31.3 & 25.4 & 36.6 & 27.3 & 21.9 & 15.8& 26.4 & 17.3&25.25 \\
    GLUNet-Semantic~\cite{truong2020glu} & 18.5 & 14.4 & 22.4 & 15.6 & 8.7 & 5.6 & 12.8 & 7.8&13.22 \\
    WarpC-GLUNet-Semantic~\cite{truong2021warp} & 27.5 & 21.4 & 36.8 & 25.7 & 18.5 & 11.9 & 28.3 & 17.6&23.46 \\
    
     \hline

    \end{tabular}
\end{table*}



</details>



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
