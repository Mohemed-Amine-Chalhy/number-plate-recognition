def compute_iou(box1, box2):
    """
    Compute IoU between two boxes: [x1, y1, x2, y2]
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0



def plate_to_track_id(plate_results, tracks, iou_tresh = 0):
    association = {}
    
    
    for track in tracks:
        x1_tr, y1_tr, x2_tr, y2_tr, track_id = track
        vehicle_box = [x1_tr, y1_tr, x2_tr, y2_tr]
        best_iou = iou_tresh
        best_plate = None
        
        for plate in plate_results:
            plate_box = plate[:4].tolist()
            iou = compute_iou(plate_box, vehicle_box)
            print(iou)
            if iou> best_iou:
                best_iou = iou 
                best_plate = plate_box
            
            
        if best_plate is not None:
                association[int(track_id)] = best_plate
        
    return association
        
    
    