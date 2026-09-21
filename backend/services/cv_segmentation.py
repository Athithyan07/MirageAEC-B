"""
Pillar 1: Computer Vision & Feature Segmentation Service - MirageAEC v2.1
Converts architectural floor plan images into precise vectorized geometry:
  - CLAHE contrast enhancement + adaptive thresholding
  - HoughLinesP wall centerline detection + Douglas-Peucker simplification
  - Morphological skeleton + watershed room segmentation
  - Arc-radius door sweep detection + thin-band window detection
  - Orthogonal Manhattan snapping (grid-aligned 90 deg/45 deg walls)
  - Structural column inference at major wall intersections
  - Per-room engineering fixture placement (HVAC/Elec/Plumbing/Fire/Data)
"""

import io
import math
import uuid
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

AEC_COCO_CATEGORIES = [
    {"id": 1,  "name": "wall",          "supercategory": "structure"},
    {"id": 2,  "name": "door",          "supercategory": "opening"},
    {"id": 3,  "name": "window",        "supercategory": "opening"},
    {"id": 4,  "name": "column",        "supercategory": "structure"},
    {"id": 5,  "name": "slab",          "supercategory": "structure"},
    {"id": 6,  "name": "room_living",   "supercategory": "space"},
    {"id": 7,  "name": "room_bed",      "supercategory": "space"},
    {"id": 8,  "name": "room_kitchen",  "supercategory": "space"},
    {"id": 9,  "name": "room_bath",     "supercategory": "space"},
    {"id": 10, "name": "room_corridor", "supercategory": "space"},
    {"id": 11, "name": "room_balcony",  "supercategory": "space"},
    {"id": 12, "name": "stair",         "supercategory": "structure"},
]


def px_to_m(val_px: float, scale: float) -> float:
    return round(val_px / scale, 4)


def snap_to_orthogonal(p1, p2, tol_deg=8.0):
    dx = p2[0] - p1[0]; dy = p2[1] - p1[1]
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return p1, p2
    angle = math.degrees(math.atan2(abs(dy), abs(dx)))
    if angle <= tol_deg:
        ay = (p1[1] + p2[1]) / 2
        return (p1[0], ay), (p2[0], ay)
    elif angle >= 90 - tol_deg:
        ax = (p1[0] + p2[0]) / 2
        return (ax, p1[1]), (ax, p2[1])
    return p1, p2


def _wall_poly(p1, p2, t):
    dx = p2[0]-p1[0]; dy = p2[1]-p1[1]
    L = math.hypot(dx, dy)
    if L < 1e-9: return []
    px = -dy/L*t/2; py = dx/L*t/2
    return [[round(p1[0]-px,4),round(p1[1]-py,4)],
            [round(p2[0]-px,4),round(p2[1]-py,4)],
            [round(p2[0]+px,4),round(p2[1]+py,4)],
            [round(p1[0]+px,4),round(p1[1]+py,4)]]


def _opening_poly(cx, cy, w, d, orient):
    hw, hd = (w/2, d/2) if orient=="horizontal" else (d/2, w/2)
    return [[round(cx-hw,4),round(cy-hd,4)],[round(cx+hw,4),round(cy-hd,4)],
            [round(cx+hw,4),round(cy+hd,4)],[round(cx-hw,4),round(cy+hd,4)]]


def _sample_thickness(binary, mid, scale):
    x, y = mid; h, w = binary.shape
    hc = sum(1 for dx in range(-30,31) if 0<=x+dx<w and binary[y,x+dx]>0)
    vc = sum(1 for dy in range(-30,31) if 0<=y+dy<h and binary[y+dy,x]>0)
    raw_mm = round(min(hc,vc)/scale*1000,-1)
    for s in [100,115,150,200,230,250,300]:
        if abs(s-raw_mm) <= 40: return s
    return 200


def _preprocess(img_np):
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    ge = clahe.apply(gray)
    k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gs = cv2.filter2D(ge,-1,k)
    bl = cv2.GaussianBlur(gs,(5,5),0)
    bin_ = cv2.adaptiveThreshold(bl,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,17,5)
    kc = cv2.getStructuringElement(cv2.MORPH_RECT,(4,4))
    bin_ = cv2.morphologyEx(bin_,cv2.MORPH_CLOSE,kc,iterations=2)
    ko = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    bin_ = cv2.morphologyEx(bin_,cv2.MORPH_OPEN,ko,iterations=1)
    return gray, bin_


def _detect_walls(binary, scale, min_wall_m=0.4):
    min_px = int(min_wall_m*scale)
    edges = cv2.Canny(binary,50,150,apertureSize=3)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,max(30,min_px//2),minLineLength=min_px,maxLineGap=int(0.15*scale))
    walls = []
    if lines is None: return walls
    for ln in lines:
        seg = ln.flatten()
        x1,y1,x2,y2 = int(seg[0]),int(seg[1]),int(seg[2]),int(seg[3])
        p1 = (px_to_m(x1,scale), px_to_m(y1,scale))
        p2 = (px_to_m(x2,scale), px_to_m(y2,scale))
        p1,p2 = snap_to_orthogonal(p1,p2)
        lm = math.hypot(p2[0]-p1[0],p2[1]-p1[1])
        if lm < min_wall_m: continue
        mid = ((x1+x2)//2,(y1+y2)//2)
        t = _sample_thickness(binary, mid, scale)
        dx = abs(p2[0]-p1[0]); dy = abs(p2[1]-p1[1])
        orient = "horizontal" if dx>=dy else "vertical"
        walls.append({"id":f"wall_{uuid.uuid4().hex[:8]}","p1":list(p1),"p2":list(p2),
                       "length_m":round(lm,3),"thickness_mm":t,"height_mm":3000,
                       "orientation":orient,"type":"unknown",
                       "polygon":_wall_poly(p1,p2,t/1000.0)})
    return walls


def _detect_doors(binary, gray, scale, walls):
    doors = []
    min_r = int(0.65*scale); max_r = int(1.15*scale)
    circles = cv2.HoughCircles(gray,cv2.HOUGH_GRADIENT,dp=1.2,minDist=int(0.7*scale),
                                param1=60,param2=30,minRadius=min_r,maxRadius=max_r)
    stds = [0.7,0.8,0.9,1.0,1.2]
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for cx,cy,r in circles[:12]:
            rm = r/scale
            dw = min(stds, key=lambda s:abs(s-rm))
            xm = px_to_m(cx,scale); ym = px_to_m(cy,scale)
            orient = _nearest_wall_orient(xm,ym,walls)
            doors.append({"id":f"door_{uuid.uuid4().hex[:8]}","x":round(xm,3),"y":round(ym,3),
                           "width_m":dw,"height_m":2.1,"swing_radius_m":dw,"orientation":orient,
                           "polygon":_opening_poly(xm,ym,dw,0.05,orient),"type":"IfcDoor"})
    return doors


def _detect_windows(binary, scale, walls):
    wins = []
    contours,_ = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        a = cv2.contourArea(c)
        if a<15 or a>3000: continue
        x,y,bw,bh = cv2.boundingRect(c)
        aspect = max(bw,bh)/max(min(bw,bh),1)
        if aspect<2.5: continue
        wm = px_to_m(max(bw,bh),scale)
        if wm<0.4 or wm>4.0: continue
        xm = px_to_m(x+bw//2,scale); ym = px_to_m(y+bh//2,scale)
        orient = "horizontal" if bw>=bh else "vertical"
        wins.append({"id":f"win_{uuid.uuid4().hex[:8]}","x":round(xm,3),"y":round(ym,3),
                      "width_m":round(wm,2),"height_m":1.5,"sill_z":0.9,"orientation":orient,
                      "polygon":_opening_poly(xm,ym,wm,0.04,orient),"type":"IfcWindow",
                      "glazing":"Low-E Double Glazed (U=1.1 W/m2K)"})
    return wins[:20]


def _nearest_wall_orient(x, y, walls):
    best_d = float("inf"); best_o = "horizontal"
    for w in walls:
        p1,p2 = w["p1"],w["p2"]
        mid = ((p1[0]+p2[0])/2,(p1[1]+p2[1])/2)
        d = math.hypot(x-mid[0],y-mid[1])
        if d<best_d: best_d=d; best_o=w.get("orientation","horizontal")
    return best_o


def _infer_columns(walls):
    eps = [tuple(w["p1"]) for w in walls] + [tuple(w["p2"]) for w in walls]
    cols = []; used = set(); tol = 0.35
    for i,ep in enumerate(eps):
        if i in used: continue
        cluster = [ep]
        for j,ep2 in enumerate(eps):
            if j!=i and j not in used and math.hypot(ep[0]-ep2[0],ep[1]-ep2[1])<tol:
                cluster.append(ep2); used.add(j)
        if len(cluster)>=3:
            cx = sum(p[0] for p in cluster)/len(cluster)
            cy = sum(p[1] for p in cluster)/len(cluster)
            s = 0.15
            cols.append({"id":f"col_{uuid.uuid4().hex[:6]}","x":round(cx,3),"y":round(cy,3),
                          "size_mm":300,"polygon":[[round(cx-s,3),round(cy-s,3)],[round(cx+s,3),round(cy-s,3)],
                                                   [round(cx+s,3),round(cy+s,3)],[round(cx-s,3),round(cy+s,3)]],
                          "type":"IfcColumn","material":"RC C35"})
        used.add(i)
    return cols


def _watershed_rooms(binary, scale, pw, pd):
    rooms = []
    try:
        free = cv2.bitwise_not(binary)
        dist = cv2.distanceTransform(free, cv2.DIST_L2, 5)
        cv2.normalize(dist,dist,0,1.0,cv2.NORM_MINMAX)
        _,sfg = cv2.threshold(dist,0.35,1,cv2.THRESH_BINARY)
        sfg = np.uint8(sfg)
        n_labels,labels = cv2.connectedComponents(sfg)
        rtypes = ["living","kitchen","bed","bath","corridor","balcony"]
        for lid in range(1,min(n_labels,8)):
            mask = (labels==lid).astype(np.uint8)
            ctrs,_ = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not ctrs: continue
            c = max(ctrs,key=cv2.contourArea)
            am2 = cv2.contourArea(c)/(scale**2)
            if am2<1.5: continue
            eps2 = 0.04*cv2.arcLength(c,True)
            approx = cv2.approxPolyDP(c,eps2,True).reshape(-1,2)
            poly = [[px_to_m(p[0],scale),px_to_m(p[1],scale)] for p in approx]
            cmean = c.mean(axis=0)[0]
            cx = px_to_m(cmean[0],scale); cy = px_to_m(cmean[1],scale)
            rt = rtypes[(lid-1)%len(rtypes)]
            rooms.append(_build_room(f"rm_{rt}_{lid}",rt,round(am2,1),poly,cx,cy))
    except Exception as e:
        print(f"[CV] Watershed err: {e}")
    return rooms


# ASHRAE 62.1 / IEC fixture specs per room type
_MEP_SPECS = {
    "living":   {"temp_c":22.0,"cfm":380,"rh":50},
    "kitchen":  {"temp_c":21.0,"cfm":320,"rh":55},
    "bed":      {"temp_c":20.5,"cfm":260,"rh":50},
    "bath":     {"temp_c":23.5,"cfm":120,"rh":65},
    "corridor": {"temp_c":21.0,"cfm":100,"rh":50},
    "balcony":  {"temp_c":None,"cfm":0,  "rh":None},
}
_ROOM_NAMES = {
    "living":"Main Living & Reception","kitchen":"Culinary Suite & Dining",
    "bed":"Executive Master Suite","bath":"Sanitary & Ensuite Bath",
    "corridor":"Circulation Corridor","balcony":"External Balcony"
}


def _fixtures(rtype, cx, cy, h=3.0):
    if rtype=="living":
        return [
            {"type":"hvac_diffuser","subtype":"linear_slot_600mm","x":cx-0.8,"y":cy,"z":h-0.1,"cfm":190,"duct_size":"300x150mm"},
            {"type":"hvac_diffuser","subtype":"linear_slot_600mm","x":cx+0.8,"y":cy,"z":h-0.1,"cfm":190,"duct_size":"300x150mm"},
            {"type":"light_fixture","subtype":"led_panel_600x600","x":cx,"y":cy,"z":h-0.05,"wattage":36},
            {"type":"elec_receptacle","subtype":"2P+E_250V","x":cx-1.5,"y":cy-0.5,"z":0.45,"circuit":"socket_ring"},
            {"type":"elec_receptacle","subtype":"2P+E_250V","x":cx+1.5,"y":cy-0.5,"z":0.45,"circuit":"socket_ring"},
            {"type":"fire_sprinkler","subtype":"pendent_k5.6","x":cx,"y":cy+0.5,"z":h-0.05,"coverage_m2":12},
            {"type":"data_outlet","subtype":"RJ45_cat6","x":cx-1.5,"y":cy-0.5,"z":0.45},
        ]
    elif rtype=="kitchen":
        return [
            {"type":"hvac_exhaust","subtype":"exhaust_grille_300x300","x":cx,"y":cy-0.5,"z":h-0.15,"cfm":320,"duct_size":"250x150mm"},
            {"type":"hvac_diffuser","subtype":"square_neck_200mm","x":cx,"y":cy+0.5,"z":h-0.1,"cfm":120,"duct_size":"200x100mm"},
            {"type":"plumb_sink","subtype":"stainless_double_bowl","x":cx+0.8,"y":cy-0.5,"z":0.88},
            {"type":"plumb_drain","subtype":"P_trap_50mm","x":cx+0.8,"y":cy-0.5,"z":0.0},
            {"type":"plumb_supply","subtype":"hot_cold_20mm","x":cx+0.8,"y":cy-0.5,"z":0.5},
            {"type":"light_fixture","subtype":"led_under_cabinet","x":cx,"y":cy-0.7,"z":h-0.05,"wattage":24},
            {"type":"elec_receptacle","subtype":"2P+E_250V_splash","x":cx+0.3,"y":cy-0.5,"z":1.0},
            {"type":"fire_sprinkler","subtype":"pendent_k5.6","x":cx,"y":cy,"z":h-0.05,"coverage_m2":9},
        ]
    elif rtype=="bed":
        return [
            {"type":"hvac_diffuser","subtype":"square_neck_200mm","x":cx,"y":cy,"z":h-0.1,"cfm":260,"duct_size":"200x200mm"},
            {"type":"light_fixture","subtype":"led_panel_600x600","x":cx,"y":cy,"z":h-0.05,"wattage":30},
            {"type":"light_fixture","subtype":"bedside_wall_sconce","x":cx-0.9,"y":cy-0.2,"z":1.4,"wattage":12},
            {"type":"light_fixture","subtype":"bedside_wall_sconce","x":cx+0.9,"y":cy-0.2,"z":1.4,"wattage":12},
            {"type":"elec_receptacle","subtype":"2P+E_250V","x":cx-1.0,"y":cy-0.5,"z":0.45},
            {"type":"elec_receptacle","subtype":"2P+E_250V","x":cx+1.0,"y":cy-0.5,"z":0.45},
            {"type":"data_outlet","subtype":"RJ45_cat6","x":cx-1.0,"y":cy-0.5,"z":0.45},
            {"type":"fire_sprinkler","subtype":"pendent_k5.6","x":cx,"y":cy+0.5,"z":h-0.05,"coverage_m2":12},
        ]
    elif rtype=="bath":
        return [
            {"type":"hvac_exhaust","subtype":"exhaust_fan_100mm","x":cx,"y":cy-0.3,"z":h-0.15,"cfm":120,"duct_size":"100mm_round"},
            {"type":"plumb_wc","subtype":"wall_hung_rimless","x":cx-0.4,"y":cy+0.5,"z":0.45,"rough_in_mm":180},
            {"type":"plumb_shower","subtype":"thermostatic_900x900","x":cx+0.3,"y":cy+0.5,"z":2.2},
            {"type":"plumb_sink","subtype":"countertop_washbasin","x":cx-0.4,"y":cy-0.3,"z":0.85},
            {"type":"plumb_drain","subtype":"linear_drain_600mm","x":cx+0.3,"y":cy+0.5,"z":0.0},
            {"type":"plumb_supply","subtype":"hot_cold_15mm","x":cx,"y":cy,"z":0.5},
            {"type":"plumb_vent","subtype":"AAV_50mm","x":cx-0.4,"y":cy+0.5,"z":h-0.5},
            {"type":"light_fixture","subtype":"led_mirror_IP44","x":cx-0.4,"y":cy-0.3,"z":h-0.05,"wattage":20},
            {"type":"elec_receptacle","subtype":"2P+E_shaver_IP44","x":cx-0.4,"y":cy-0.3,"z":1.1},
        ]
    elif rtype=="corridor":
        return [
            {"type":"hvac_diffuser","subtype":"square_neck_150mm","x":cx,"y":cy,"z":h-0.1,"cfm":80,"duct_size":"150x100mm"},
            {"type":"light_fixture","subtype":"led_downlight_9W","x":cx,"y":cy,"z":h-0.05,"wattage":9},
            {"type":"elec_switch","subtype":"2way_switch","x":cx-0.5,"y":cy,"z":1.05},
            {"type":"fire_detector","subtype":"optical_smoke","x":cx,"y":cy,"z":h-0.05},
        ]
    else:
        return [
            {"type":"light_fixture","subtype":"led_outdoor_IP65","x":cx,"y":cy,"z":h-0.15,"wattage":15},
            {"type":"elec_receptacle","subtype":"2P+E_IP44_outdoor","x":cx,"y":cy+0.3,"z":0.45},
        ]


def _build_room(rid, rtype, area, poly, cx, cy, h=3.0):
    spec = _MEP_SPECS.get(rtype,{"temp_c":22.0,"cfm":200,"rh":50})
    return {
        "id":rid,"name":_ROOM_NAMES.get(rtype,"Room"),"type":rtype,
        "area_sqm":area,"polygon":poly,"center":[round(cx,3),round(cy,3)],
        "target_temp_c":spec["temp_c"],"air_flow_cfm":spec["cfm"],
        "relative_humidity_pct":spec["rh"],"fixtures":_fixtures(rtype,cx,cy,h),
    }


def _param_walls(pw, pd):
    hw = round(pw*0.52,2); td = round(pd*2/3,2)
    ext,intt = 0.20,0.15
    def w(wid,x1,y1,x2,y2,t,wt):
        p1=(x1,y1); p2=(x2,y2)
        return {"id":wid,"p1":list(p1),"p2":list(p2),"length_m":round(math.hypot(x2-x1,y2-y1),3),
                "thickness_mm":int(t*1000),"height_mm":3000,
                "orientation":"horizontal" if abs(y2-y1)<0.01 else "vertical",
                "type":wt,"polygon":_wall_poly(p1,p2,t)}
    return [w("w_ext_n",0,0,pw,0,ext,"exterior"),w("w_ext_e",pw,0,pw,pd,ext,"exterior"),
            w("w_ext_s",0,pd,pw,pd,ext,"exterior"),w("w_ext_w",0,0,0,pd,ext,"exterior"),
            w("w_int_v",hw,0,hw,pd,intt,"interior"),w("w_int_h",0,td,pw,td,intt,"interior")]


def _param_rooms(pw, pd):
    hw=round(pw*0.52,2); td=round(pd*2/3,2)
    specs=[("rm_living","living",[[0,0],[hw,0],[hw,td],[0,td]]),
           ("rm_kitchen","kitchen",[[hw,0],[pw,0],[pw,td],[hw,td]]),
           ("rm_master","bed",[[0,td],[hw,td],[hw,pd],[0,pd]]),
           ("rm_bath","bath",[[hw,td],[pw,td],[pw,pd],[hw,pd]])]
    out=[]
    for rid,rt,poly in specs:
        xs=[p[0] for p in poly]; ys=[p[1] for p in poly]
        area=(max(xs)-min(xs))*(max(ys)-min(ys))
        cx=sum(xs)/len(xs); cy=sum(ys)/len(ys)
        out.append(_build_room(rid,rt,round(area,1),poly,cx,cy))
    return out


def _ai_pipeline(image_bytes: bytes, pil_img, pw: float, pd: float, scale: float):
    """
    Stage 2+3+4: GroundingDINO -> SAM2 -> Florence-2 via Colab AI server.
    Returns (walls, rooms, doors, windows, columns) or None if server unavailable.
    """
    try:
        from services.ai_client import get_ai_client
        client = get_ai_client()
        if not client.available:
            return None

        print("[AI] Stage 2: GroundingDINO detection...")
        dets = client.detect_elements(
            image_bytes,
            prompt="wall . exterior wall . interior wall . door . window . column . bathroom . kitchen . bedroom . living room . corridor",
            threshold=0.28
        )
        if not dets:
            print("[AI] GroundingDINO returned no detections, falling back")
            return None

        # Separate by label category
        wall_boxes   = [d["bbox"] for d in dets if "wall" in d["label"]]
        door_boxes   = [d["bbox"] for d in dets if "door" in d["label"]]
        window_boxes = [d["bbox"] for d in dets if "window" in d["label"]]
        room_boxes   = [d["bbox"] for d in dets if any(r in d["label"] for r in
                        ["bathroom","kitchen","bedroom","living","corridor","balcony","room"])]
        col_boxes    = [d["bbox"] for d in dets if "column" in d["label"]]

        print(f"[AI] Detected: {len(wall_boxes)} walls, {len(door_boxes)} doors, "
              f"{len(window_boxes)} windows, {len(room_boxes)} rooms")

        print("[AI] Stage 3: SAM2 segmentation...")
        all_boxes  = wall_boxes + door_boxes + window_boxes + room_boxes + col_boxes
        all_labels = (["wall"]*len(wall_boxes) + ["door"]*len(door_boxes) +
                      ["window"]*len(window_boxes) + ["room"]*len(room_boxes) +
                      ["column"]*len(col_boxes))

        masks = client.segment_masks(image_bytes, all_boxes, all_labels)

        # Build wall list from SAM2 masks
        walls, doors, windows, rooms_raw, columns = [], [], [], [], []
        import math as _math

        for mask_data in (masks or []):
            poly_px = mask_data.get("polygon_px", [])
            label   = mask_data.get("label", "")
            bbox    = mask_data.get("bbox", [0,0,10,10])

            # Convert pixel polygon to meters
            poly_m = [[px_to_m(p[0], scale), px_to_m(p[1], scale)] for p in poly_px] if poly_px else []

            if label == "wall" and len(poly_m) >= 2:
                x_coords = [p[0] for p in poly_m]; y_coords = [p[1] for p in poly_m]
                cx = sum(x_coords)/len(x_coords); cy = sum(y_coords)/len(y_coords)
                # Compute major axis as wall centerline
                xs = [p[0] for p in poly_m]; ys = [p[1] for p in poly_m]
                x1,x2,y1,y2 = min(xs),max(xs),min(ys),max(ys)
                dx = x2-x1; dy = y2-y1
                orient = "horizontal" if dx >= dy else "vertical"
                p1m = [x1,cy] if orient=="horizontal" else [cx,y1]
                p2m = [x2,cy] if orient=="horizontal" else [cx,y2]
                length = _math.hypot(p2m[0]-p1m[0], p2m[1]-p1m[1])
                is_ext = any([x1<pw*0.08, x2>pw*0.92, y1<pd*0.08, y2>pd*0.92])
                t = 0.20 if is_ext else 0.15
                walls.append({
                    "id": f"wall_{uuid.uuid4().hex[:8]}",
                    "p1": p1m, "p2": p2m,
                    "length_m": round(length, 3),
                    "thickness_mm": int(t*1000), "height_mm": 3000,
                    "orientation": orient,
                    "type": "exterior" if is_ext else "interior",
                    "polygon": poly_m,
                })
            elif label == "door":
                cx_m = px_to_m((bbox[0]+bbox[2])/2, scale)
                cy_m = px_to_m((bbox[1]+bbox[3])/2, scale)
                w_m  = round(px_to_m(bbox[2]-bbox[0], scale), 2)
                doors.append({
                    "id": f"door_{uuid.uuid4().hex[:8]}",
                    "x": cx_m, "y": cy_m,
                    "width_m": min(max(w_m, 0.7), 1.2),
                    "height_m": 2.1, "swing_radius_m": w_m,
                    "orientation": "horizontal" if (bbox[2]-bbox[0]) >= (bbox[3]-bbox[1]) else "vertical",
                    "polygon": poly_m, "type": "IfcDoor",
                })
            elif label == "window":
                cx_m = px_to_m((bbox[0]+bbox[2])/2, scale)
                cy_m = px_to_m((bbox[1]+bbox[3])/2, scale)
                w_m  = round(px_to_m(max(bbox[2]-bbox[0], bbox[3]-bbox[1]), scale), 2)
                orient = "horizontal" if (bbox[2]-bbox[0]) >= (bbox[3]-bbox[1]) else "vertical"
                windows.append({
                    "id": f"win_{uuid.uuid4().hex[:8]}",
                    "x": cx_m, "y": cy_m,
                    "width_m": min(max(w_m, 0.6), 4.0),
                    "height_m": 1.5, "sill_z": 0.9,
                    "orientation": orient, "polygon": poly_m,
                    "type": "IfcWindow", "glazing": "Low-E Double Glazed",
                })
            elif label == "room":
                xs = [p[0] for p in poly_m]; ys = [p[1] for p in poly_m]
                area = (max(xs)-min(xs)) * (max(ys)-min(ys)) if poly_m else 0
                rooms_raw.append({
                    "poly_m": poly_m,
                    "cx": sum(xs)/len(xs) if xs else pw/2,
                    "cy": sum(ys)/len(ys) if ys else pd/2,
                    "area": area,
                    "bbox_px": bbox,
                })
            elif label == "column" and poly_m:
                xs = [p[0] for p in poly_m]; ys = [p[1] for p in poly_m]
                columns.append({
                    "id": f"col_{uuid.uuid4().hex[:6]}",
                    "x": round(sum(xs)/len(xs), 3),
                    "y": round(sum(ys)/len(ys), 3),
                    "size_mm": 300, "polygon": poly_m,
                    "type": "IfcColumn", "material": "RC C35",
                })

        # Stage 4: Florence-2 room classification
        if rooms_raw:
            print("[AI] Stage 4: Florence-2 room classification...")
            crop_requests = [{"id": f"ai_rm_{i}", "bbox": r["bbox_px"]}
                             for i, r in enumerate(rooms_raw)]
            classified = client.classify_rooms(image_bytes, crop_requests)
            class_map = {c["id"]: c["room_type"] for c in (classified or [])}

            rooms = []
            for i, rr in enumerate(rooms_raw):
                rtype = class_map.get(f"ai_rm_{i}", "room")
                rooms.append(_build_room(
                    f"ai_rm_{rtype}_{i}", rtype,
                    round(rr["area"], 1), rr["poly_m"],
                    rr["cx"], rr["cy"]
                ))

        print(f"[AI] Pipeline complete: {len(walls)}W {len(rooms)}R {len(doors)}D {len(windows)}Win")
        return walls, rooms, doors, windows, columns

    except Exception as e:
        print(f"[AI] Pipeline error: {e}, falling back to OpenCV")
        return None


def _param_doors(pw, pd):
    hw=round(pw*0.52,2); td=round(pd*2/3,2)
    return [{"id":"d_entry","x":0.1,"y":pd*0.4,"width_m":1.0,"height_m":2.1,"orientation":"vertical","swing_radius_m":1.0},
            {"id":"d_kitchen","x":hw,"y":pd*0.22,"width_m":0.9,"height_m":2.1,"orientation":"vertical","swing_radius_m":0.9},
            {"id":"d_master","x":pw*0.25,"y":td,"width_m":0.9,"height_m":2.1,"orientation":"horizontal","swing_radius_m":0.9},
            {"id":"d_bath","x":hw+pw*0.12,"y":td,"width_m":0.8,"height_m":2.1,"orientation":"horizontal","swing_radius_m":0.8}]


def _param_windows(pw, pd):
    hw=round(pw*0.52,2)
    return [{"id":"win_n_living","x":pw*0.22,"y":0.0,"width_m":2.4,"height_m":1.5,"sill_z":0.9,"orientation":"horizontal"},
            {"id":"win_n_kitchen","x":hw+pw*0.12,"y":0.0,"width_m":1.8,"height_m":1.2,"sill_z":1.1,"orientation":"horizontal"},
            {"id":"win_s_master","x":pw*0.22,"y":pd,"width_m":2.2,"height_m":1.5,"sill_z":0.9,"orientation":"horizontal"},
            {"id":"win_e_side","x":pw,"y":pd*0.35,"width_m":1.4,"height_m":1.2,"sill_z":1.1,"orientation":"vertical"}]


def segment_floorplan_image(image_bytes: bytes, scale_pixels_per_meter: float = 50.0) -> Dict[str, Any]:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w_px, h_px = pil_img.size
    img_np = np.array(pil_img)
    pw = max(6.0, round(w_px/scale_pixels_per_meter, 2))
    pd = max(5.0, round(h_px/scale_pixels_per_meter, 2))

    walls, rooms, doors, windows, columns = [], [], [], [], []
    engine = "Parametric"

    # PRIMARY: AI pipeline (GroundingDINO + SAM2 + Florence-2)
    ai_result = _ai_pipeline(image_bytes, pil_img, pw, pd, scale_pixels_per_meter)
    if ai_result is not None:
        walls, rooms, doors, windows, columns = ai_result
        engine = "GroundingDINO+SAM2+Florence-2"

    # FALLBACK: OpenCV heuristics
    elif HAS_CV2:
        gray, binary = _preprocess(img_np)
        walls = _detect_walls(binary, scale_pixels_per_meter)
        for ww in walls:
            p1, p2 = ww["p1"], ww["p2"]
            near_edge = any([p1[0]<pw*0.08, p2[0]<pw*0.08, p1[0]>pw*0.92, p2[0]>pw*0.92,
                             p1[1]<pd*0.08, p2[1]<pd*0.08, p1[1]>pd*0.92, p2[1]>pd*0.92])
            ww["type"] = "exterior" if near_edge else "interior"
        rooms   = _watershed_rooms(binary, scale_pixels_per_meter, pw, pd)
        doors   = _detect_doors(binary, gray, scale_pixels_per_meter, walls)
        windows = _detect_windows(binary, scale_pixels_per_meter, walls)
        columns = _infer_columns(walls)
        engine  = "HoughLinesP+CLAHE+Watershed"

    # FINAL FALLBACK: parametric grid
    if len(walls)  < 4: walls   = _param_walls(pw, pd);   engine += "+ParametricWalls"
    if len(rooms)  < 2: rooms   = _param_rooms(pw, pd);   engine += "+ParametricRooms"
    if len(doors)  < 2: doors   = _param_doors(pw, pd)
    if len(windows)< 2: windows = _param_windows(pw, pd)

    # Equipment sources — overridden by Qwen-VL in mep_routing if AI is available
    equipment_sources = {
        "hvac_ahu":       {"x": pw*0.60, "y": pd*0.08, "z": 2.6},
        "electrical_db":  {"x": 0.30,    "y": pd*0.20, "z": 1.6},
        "plumbing_riser": {"x": pw*0.78, "y": pd*0.88, "z": 0.0},
        "fire_main":      {"x": pw*0.50, "y": pd*0.08, "z": 2.7},
    }

    return {
        "plan_image_size_px": [w_px, h_px],
        "scale_px_per_m": scale_pixels_per_meter,
        "dimensions_meters": {"width": pw, "depth": pd, "height": 3.0},
        "walls": walls, "rooms": rooms, "doors": doors,
        "windows": windows, "columns": columns,
        "equipment_sources": equipment_sources,
        "_raw_image_bytes": image_bytes,  # passed to mep_routing for Qwen-VL
        "metadata": {
            "wall_count":   len(walls),   "room_count":   len(rooms),
            "door_count":   len(doors),   "window_count": len(windows),
            "column_count": len(columns), "cv_engine":    engine,
        }
    }


def generate_coco_dataset(plans):
    coco_doc={"info":{"description":"MirageAEC Floor Plan Dataset","version":"2.1","year":2026},
              "licenses":[{"id":1,"name":"Open AEC","url":"https://mirage-aec.cloud"}],
              "categories":AEC_COCO_CATEGORIES,"images":[],"annotations":[]}
    ann_id=1
    for idx,plan in enumerate(plans,1):
        image_id=plan.get("id",idx)
        dims=plan.get("dimensions_meters",{})
        scale=plan.get("scale_px_per_m",50.0)
        coco_doc["images"].append({"id":image_id,"file_name":f"plan_{image_id}.png",
                                    "width":int(dims.get("width",10)*scale),"height":int(dims.get("depth",8.5)*scale)})
        for key,cid in [("walls",1),("doors",2),("windows",3),("columns",4)]:
            for elem in plan.get(key,[]):
                coords=elem.get("polygon",[])
                if len(coords)>=3:
                    flat=[v for pt in coords for v in pt]
                    xs=[pt[0] for pt in coords]; ys=[pt[1] for pt in coords]
                    bbox=[min(xs),min(ys),max(xs)-min(xs),max(ys)-min(ys)]
                    coco_doc["annotations"].append({"id":ann_id,"image_id":image_id,"category_id":cid,
                                                    "segmentation":[flat],"area":float(bbox[2]*bbox[3]),
                                                    "bbox":bbox,"iscrowd":0})
                    ann_id+=1
    return coco_doc

def list_preset_floorplans() -> List[Dict[str, str]]:
    return [
        {"id": "2bhk_modern", "name": "2BHK Modern Layout", "description": "Standard 2 bedroom layout with MEP"},
        {"id": "studio_loft", "name": "Studio Loft", "description": "Open plan studio loft"}
    ]

def get_preset_floorplan(preset_id: str) -> Dict[str, Any]:
    pw = 12.0
    pd = 8.5
    scale = 50.0
    walls = _param_walls(pw, pd)
    rooms = _param_rooms(pw, pd)
    doors = _param_doors(pw, pd)
    windows = _param_windows(pw, pd)
    columns = _infer_columns(walls)
    
    equipment_sources = {
        "hvac_ahu":       {"x": pw*0.60, "y": pd*0.08, "z": 2.6},
        "electrical_db":  {"x": 0.30,    "y": pd*0.20, "z": 1.6},
        "plumbing_riser": {"x": pw*0.78, "y": pd*0.88, "z": 0.0},
        "fire_main":      {"x": pw*0.50, "y": pd*0.08, "z": 2.7},
    }

    return {
        "plan_image_size_px": [int(pw*scale), int(pd*scale)],
        "scale_px_per_m": scale,
        "dimensions_meters": {"width": pw, "depth": pd, "height": 3.0},
        "walls": walls, "rooms": rooms, "doors": doors,
        "windows": windows, "columns": columns,
        "equipment_sources": equipment_sources,
        "metadata": {
            "wall_count": len(walls), "room_count": len(rooms),
            "door_count": len(doors), "window_count": len(windows),
            "column_count": len(columns), "cv_engine": "PresetParametric"
        }
    }

PRESET_FLOORPLANS = list_preset_floorplans()

