""" This file contains the inference code for a mmpose detection model. """

from tqdm import tqdm
from mmpose.apis import inference_topdown


def pose_estimation(model, dataset):
    for i in tqdm(range(len(dataset))):
        inputs, meta_info = dataset[i]
        img = inputs["img"]
        bboxes = inputs["bboxes"]
        
        predictions = []
        for i, bbox in enumerate(bboxes):
            result = inference_topdown(model=model, img=img, bboxes=[bbox], bbox_format=dataset.bbox_format)[0]
            predictions.append(result.pred_instances)
        
        dataset.save_prediction(inputs=inputs, meta_info=meta_info, predictions=predictions)
