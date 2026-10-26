"""
Helper script to generate high-resolution sample room photographs for testing
and viva demonstration.
"""
import os
import numpy as np
import cv2

def create_sample_bedroom(output_path):
    # 1280x800 modern perspective bedroom scene
    h, w = 800, 1280
    img = np.ones((h, w, 3), dtype=np.uint8) * 240 # Warm light walls

    # Vanishing point near center-right
    vx, vy = int(w * 0.52), int(h * 0.42)

    # 1. Back wall (rectangle)
    bw_x1, bw_y1 = int(w * 0.28), int(h * 0.18)
    bw_x2, bw_y2 = int(w * 0.76), int(h * 0.65)
    cv2.rectangle(img, (bw_x1, bw_y1), (bw_x2, bw_y2), (230, 235, 238), -1)
    cv2.rectangle(img, (bw_x1, bw_y1), (bw_x2, bw_y2), (180, 190, 195), 2)

    # 2. Ceiling (polygon from top to back wall)
    ceiling_pts = np.array([[0, 0], [w, 0], [bw_x2, bw_y1], [bw_x1, bw_y1]], np.int32)
    cv2.fillPoly(img, [ceiling_pts], (245, 248, 250))
    cv2.polylines(img, [ceiling_pts], True, (190, 195, 200), 2)

    # 3. Floor (polygon from bottom to back wall) with hardwood plank perspective lines
    floor_pts = np.array([[0, h], [w, h], [bw_x2, bw_y2], [bw_x1, bw_y2]], np.int32)
    cv2.fillPoly(img, [floor_pts], (140, 170, 200)) # Warm oak floor tone (BGR)
    cv2.polylines(img, [floor_pts], True, (110, 140, 170), 2)

    # Hardwood floor perspective planks
    for fx in range(-200, w + 400, 90):
        cv2.line(img, (fx, h), (int(bw_x1 + (fx - bw_x1) * 0.25), bw_y2), (120, 150, 180), 1)

    # 4. Left wall
    left_wall_pts = np.array([[0, 0], [bw_x1, bw_y1], [bw_x1, bw_y2], [0, h]], np.int32)
    cv2.fillPoly(img, [left_wall_pts], (225, 230, 235))
    cv2.polylines(img, [left_wall_pts], True, (180, 185, 190), 2)

    # 5. Right wall
    right_wall_pts = np.array([[w, 0], [bw_x2, bw_y1], [bw_x2, bw_y2], [w, h]], np.int32)
    cv2.fillPoly(img, [right_wall_pts], (215, 220, 225))
    cv2.polylines(img, [right_wall_pts], True, (175, 180, 185), 2)

    # 6. Window on back wall
    win_x1, win_y1 = int(bw_x1 + (bw_x2 - bw_x1) * 0.55), int(bw_y1 + (bw_y2 - bw_y1) * 0.12)
    win_x2, win_y2 = int(bw_x1 + (bw_x2 - bw_x1) * 0.90), int(bw_y1 + (bw_y2 - bw_y1) * 0.70)
    cv2.rectangle(img, (win_x1, win_y1), (win_x2, win_y2), (240, 225, 200), -1) # Sky tint
    cv2.rectangle(img, (win_x1, win_y1), (win_x2, win_y2), (90, 100, 110), 3)
    # Window mullions
    win_mid_x = (win_x1 + win_x2) // 2
    win_mid_y = (win_y1 + win_y2) // 2
    cv2.line(img, (win_mid_x, win_y1), (win_mid_x, win_y2), (90, 100, 110), 2)
    cv2.line(img, (win_x1, win_mid_y), (win_x2, win_mid_y), (90, 100, 110), 2)

    # 7. Bed on floor (perspective box)
    bed_front_bl = (int(w * 0.32), int(h * 0.92))
    bed_front_br = (int(w * 0.68), int(h * 0.92))
    bed_front_tl = (int(w * 0.32), int(h * 0.75))
    bed_front_tr = (int(w * 0.68), int(h * 0.75))

    bed_back_bl = (int(w * 0.36), int(h * 0.68))
    bed_back_br = (int(w * 0.64), int(h * 0.68))
    bed_back_tl = (int(w * 0.36), int(h * 0.60))
    bed_back_tr = (int(w * 0.64), int(h * 0.60))

    # Bed front face
    bed_front = np.array([bed_front_bl, bed_front_br, bed_front_tr, bed_front_tl], np.int32)
    cv2.fillPoly(img, [bed_front], (180, 120, 80)) # Warm wooden base
    cv2.polylines(img, [bed_front], True, (130, 80, 40), 2)

    # Bed top mattress & duvet
    bed_top = np.array([bed_front_tl, bed_front_tr, bed_back_tr, bed_back_tl], np.int32)
    cv2.fillPoly(img, [bed_top], (220, 205, 195)) # Soft linen
    cv2.polylines(img, [bed_top], True, (160, 145, 135), 2)

    # Headboard on back wall
    headboard_pts = np.array([
        (int(w * 0.35), int(h * 0.68)),
        (int(w * 0.65), int(h * 0.68)),
        (int(w * 0.65), int(h * 0.48)),
        (int(w * 0.35), int(h * 0.48))
    ], np.int32)
    cv2.fillPoly(img, [headboard_pts], (80, 95, 120))
    cv2.polylines(img, [headboard_pts], True, (50, 65, 90), 2)

    # Pillows
    cv2.rectangle(img, (int(w * 0.38), int(h * 0.58)), (int(w * 0.48), int(h * 0.65)), (250, 250, 250), -1)
    cv2.rectangle(img, (int(w * 0.38), int(h * 0.58)), (int(w * 0.48), int(h * 0.65)), (180, 180, 180), 1)
    cv2.rectangle(img, (int(w * 0.52), int(h * 0.58)), (int(w * 0.62), int(h * 0.65)), (250, 250, 250), -1)
    cv2.rectangle(img, (int(w * 0.52), int(h * 0.58)), (int(w * 0.62), int(h * 0.65)), (180, 180, 180), 1)

    # 8. Nightstand on left
    cv2.rectangle(img, (int(w * 0.22), int(h * 0.68)), (int(w * 0.30), int(h * 0.80)), (130, 90, 60), -1)
    cv2.rectangle(img, (int(w * 0.22), int(h * 0.68)), (int(w * 0.30), int(h * 0.80)), (90, 60, 40), 2)
    # Lamp
    cv2.line(img, (int(w * 0.26), int(h * 0.68)), (int(w * 0.26), int(h * 0.60)), (100, 100, 100), 2)
    lamp_pts = np.array([(int(w * 0.23), int(h * 0.60)), (int(w * 0.29), int(h * 0.60)), (int(w * 0.26), int(h * 0.54))], np.int32)
    cv2.fillPoly(img, [lamp_pts], (210, 230, 245))

    # 9. Door on left wall (perspective)
    door_pts = np.array([
        (int(w * 0.05), int(h * 0.82)),
        (int(w * 0.16), int(h * 0.72)),
        (int(w * 0.16), int(h * 0.25)),
        (int(w * 0.05), int(h * 0.28))
    ], np.int32)
    cv2.fillPoly(img, [door_pts], (190, 175, 160))
    cv2.polylines(img, [door_pts], True, (120, 100, 90), 2)
    # Door handle
    cv2.circle(img, (int(w * 0.14), int(h * 0.52)), 4, (50, 50, 50), -1)

    # 10. Modern Wall Art on back wall
    art_pts = np.array([
        (int(bw_x1 + 40), int(bw_y1 + 50)),
        (int(bw_x1 + 180), int(bw_y1 + 50)),
        (int(bw_x1 + 180), int(bw_y1 + 180)),
        (int(bw_x1 + 40), int(bw_y1 + 180))
    ], np.int32)
    cv2.fillPoly(img, [art_pts], (80, 130, 160))
    cv2.polylines(img, [art_pts], True, (40, 40, 40), 3)

    cv2.imwrite(output_path, img)
    print(f"Sample bedroom saved: {output_path}")

def create_sample_living_room(output_path):
    h, w = 800, 1280
    img = np.ones((h, w, 3), dtype=np.uint8) * 235

    bw_x1, bw_y1 = int(w * 0.22), int(h * 0.20)
    bw_x2, bw_y2 = int(w * 0.78), int(h * 0.64)
    cv2.rectangle(img, (bw_x1, bw_y1), (bw_x2, bw_y2), (225, 230, 235), -1)
    cv2.rectangle(img, (bw_x1, bw_y1), (bw_x2, bw_y2), (180, 185, 190), 2)

    # Ceiling & floor
    ceiling_pts = np.array([[0, 0], [w, 0], [bw_x2, bw_y1], [bw_x1, bw_y1]], np.int32)
    cv2.fillPoly(img, [ceiling_pts], (245, 245, 248))
    cv2.polylines(img, [ceiling_pts], True, (200, 200, 205), 2)

    floor_pts = np.array([[0, h], [w, h], [bw_x2, bw_y2], [bw_x1, bw_y2]], np.int32)
    cv2.fillPoly(img, [floor_pts], (190, 205, 215)) # Modern polished tile floor
    cv2.polylines(img, [floor_pts], True, (150, 165, 175), 2)

    # Walls
    left_wall_pts = np.array([[0, 0], [bw_x1, bw_y1], [bw_x1, bw_y2], [0, h]], np.int32)
    cv2.fillPoly(img, [left_wall_pts], (220, 225, 230))
    cv2.polylines(img, [left_wall_pts], True, (170, 175, 180), 2)

    right_wall_pts = np.array([[w, 0], [bw_x2, bw_y1], [bw_x2, bw_y2], [w, h]], np.int32)
    cv2.fillPoly(img, [right_wall_pts], (210, 215, 220))
    cv2.polylines(img, [right_wall_pts], True, (165, 170, 175), 2)

    # Double Window on right wall
    rwin_pts = np.array([
        (int(w * 0.82), int(h * 0.30)),
        (int(w * 0.96), int(h * 0.24)),
        (int(w * 0.96), int(h * 0.70)),
        (int(w * 0.82), int(h * 0.65))
    ], np.int32)
    cv2.fillPoly(img, [rwin_pts], (240, 230, 210))
    cv2.polylines(img, [rwin_pts], True, (80, 90, 100), 2)

    # Center modern Sofa
    sofa_bl = (int(w * 0.28), int(h * 0.88))
    sofa_br = (int(w * 0.72), int(h * 0.88))
    sofa_tl = (int(w * 0.28), int(h * 0.70))
    sofa_tr = (int(w * 0.72), int(h * 0.70))
    sofa_box = np.array([sofa_bl, sofa_br, sofa_tr, sofa_tl], np.int32)
    cv2.fillPoly(img, [sofa_box], (110, 95, 80)) # Charcoal/navy sofa
    cv2.polylines(img, [sofa_box], True, (70, 60, 50), 2)

    # Sofa backrest
    sofa_back = np.array([(int(w * 0.28), int(h * 0.70)), (int(w * 0.72), int(h * 0.70)),
                          (int(w * 0.70), int(h * 0.58)), (int(w * 0.30), int(h * 0.58))], np.int32)
    cv2.fillPoly(img, [sofa_back], (85, 75, 65))
    cv2.polylines(img, [sofa_back], True, (50, 45, 40), 2)

    # Coffee Table in front of sofa
    coffee_pts = np.array([(int(w * 0.38), int(h * 0.96)), (int(w * 0.62), int(h * 0.96)),
                           (int(w * 0.60), int(h * 0.90)), (int(w * 0.40), int(h * 0.90))], np.int32)
    cv2.fillPoly(img, [coffee_pts], (160, 140, 120))
    cv2.polylines(img, [coffee_pts], True, (100, 80, 60), 2)

    # Large TV unit on left wall
    tv_pts = np.array([
        (int(w * 0.04), int(h * 0.38)),
        (int(w * 0.17), int(h * 0.40)),
        (int(w * 0.17), int(h * 0.62)),
        (int(w * 0.04), int(h * 0.64))
    ], np.int32)
    cv2.fillPoly(img, [tv_pts], (30, 30, 35))
    cv2.polylines(img, [tv_pts], True, (70, 70, 75), 2)

    cv2.imwrite(output_path, img)
    print(f"Sample living room saved: {output_path}")

if __name__ == "__main__":
    assets_dir = r"C:\Users\simbu T\.gemini\antigravity\scratch\ai-room-reconstruction\assets"
    os.makedirs(assets_dir, exist_ok=True)
    create_sample_bedroom(os.path.join(assets_dir, "sample_bedroom.png"))
    create_sample_living_room(os.path.join(assets_dir, "sample_living_room.png"))
