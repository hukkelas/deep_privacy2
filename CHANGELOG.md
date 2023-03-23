# Changelog

## 23.03.2023
- Quality of life improvements
- Add support for refined keypoints for the FDH dataset. 
- Add FDF128 dataset loader with webdataset.
- Support for using detector and anonymizer from DeepPrivacy1.
- Update visualization of keypoints
- Fix bug for upsampling/downsampling in the anonymization pipeline.
- Support for keypoint-guided face anonymization.
- Add ViTPose + Mask-RCNN detection model for keypoint-guided full-body anonymization.
- Set caching of detections to False as default, as it can produce unexpected behaviour. For example, using a different score threshold requires re-run of detector.
- Add Gradio Demos for face and body anonymization