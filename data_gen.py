import enum
import numpy as np
from PIL import Image, ImageDraw
import os

WIDTH = 1
NUM_SAMPLES = 1000

def gen_img(top_line_length, bottom_line_length, arrow_length, arrow_angle, type="illusion"):

    top_line_pos = np.random.randint(48,109)
    bottom_line_pos = np.random.randint(148,209)

    top_sx = 112 - top_line_length // 2
    top_ex = 112 + top_line_length // 2
    bot_sx = 112 - bottom_line_length // 2
    bot_ex = 112 + bottom_line_length // 2
    
    proj_x = np.cos(arrow_angle) * arrow_length
    proj_y = np.sin(arrow_angle) * arrow_length

    img = Image.new('L', (224,224), color=255)

    draw = ImageDraw.Draw(img)

    # Top line
    draw.line([(top_sx, top_line_pos), (top_ex, top_line_pos)], fill=0, width=WIDTH)

    # Bottom line
    draw.line([(bot_sx, bottom_line_pos), (bot_ex, bottom_line_pos)], fill=0, width=WIDTH)

    # Draw arrow tips
    if type == "cross_fin":
        # top left
        draw.line([(top_sx - proj_x, top_line_pos - proj_y), (top_sx + proj_x, top_line_pos + proj_y)], fill=0, width=WIDTH)
        draw.line([(top_sx - proj_x, top_line_pos + proj_y), (top_sx + proj_x, top_line_pos - proj_y)], fill=0, width=WIDTH)

        # top right
        draw.line([(top_ex - proj_x, top_line_pos - proj_y), (top_ex + proj_x, top_line_pos + proj_y)], fill=0, width=WIDTH)
        draw.line([(top_ex - proj_x, top_line_pos + proj_y), (top_ex + proj_x, top_line_pos - proj_y)], fill=0, width=WIDTH)

        # bottom left
        draw.line([(bot_sx + proj_x, bottom_line_pos - proj_y), (bot_sx - proj_x, bottom_line_pos + proj_y)], fill=0, width=WIDTH)
        draw.line([(bot_sx + proj_x, bottom_line_pos + proj_y), (bot_sx - proj_x, bottom_line_pos - proj_y)], fill=0, width=WIDTH)

        # bottom right
        draw.line([(bot_ex + proj_x, bottom_line_pos - proj_y), (bot_ex - proj_x, bottom_line_pos + proj_y)], fill=0, width=WIDTH)
        draw.line([(bot_ex + proj_x, bottom_line_pos + proj_y), (bot_ex - proj_x, bottom_line_pos - proj_y)], fill=0, width=WIDTH)
    elif type == "control": # Control 
        # top left
        draw.line([(top_sx, top_line_pos), (top_sx + proj_x, top_line_pos + proj_y)], fill=0, width=WIDTH)
        draw.line([(top_sx, top_line_pos), (top_sx + proj_x, top_line_pos - proj_y)], fill=0, width=WIDTH)

        # top right
        draw.line([(top_ex, top_line_pos), (top_ex + proj_x, top_line_pos + proj_y)], fill=0, width=WIDTH)
        draw.line([(top_ex, top_line_pos), (top_ex + proj_x, top_line_pos - proj_y)], fill=0, width=WIDTH)

        # bottom left
        draw.line([(bot_sx, bottom_line_pos ), (bot_sx - proj_x, bottom_line_pos + proj_y)], fill=0, width=WIDTH)
        draw.line([(bot_sx, bottom_line_pos), (bot_sx - proj_x, bottom_line_pos - proj_y)], fill=0, width=WIDTH)

        # bottom right
        draw.line([(bot_ex, bottom_line_pos), (bot_ex - proj_x, bottom_line_pos + proj_y)], fill=0, width=WIDTH)
        draw.line([(bot_ex, bottom_line_pos), (bot_ex - proj_x, bottom_line_pos - proj_y)], fill=0, width=WIDTH)
    elif type == "illusion": # Illusion
        # top left
        draw.line([(top_sx, top_line_pos), (top_sx - proj_x, top_line_pos + proj_y)], fill=0, width=WIDTH)
        draw.line([(top_sx, top_line_pos), (top_sx - proj_x, top_line_pos - proj_y)], fill=0, width=WIDTH)

        # top right
        draw.line([(top_ex, top_line_pos), (top_ex + proj_x, top_line_pos + proj_y)], fill=0, width=WIDTH)
        draw.line([(top_ex, top_line_pos), (top_ex + proj_x, top_line_pos - proj_y)], fill=0, width=WIDTH)

        # bottom left
        draw.line([(bot_sx, bottom_line_pos), (bot_sx + proj_x, bottom_line_pos + proj_y)], fill=0, width=WIDTH)
        draw.line([(bot_sx, bottom_line_pos), (bot_sx + proj_x, bottom_line_pos - proj_y)], fill=0, width=WIDTH)

        # bottom right
        draw.line([(bot_ex, bottom_line_pos), (bot_ex - proj_x, bottom_line_pos + proj_y)], fill=0, width=WIDTH)
        draw.line([(bot_ex, bottom_line_pos), (bot_ex - proj_x, bottom_line_pos - proj_y)], fill=0, width=WIDTH)

    return img

if __name__ == "__main__":
    types = ["cross_fin", "control", "illusion"]
    categories = ["S", "L"]

    for i, type in enumerate(types):
        idx = 0
        for j, category in enumerate(categories):
            base_dir = "./output/"+type+"/"
            os.makedirs(base_dir, exist_ok=True)

            for k in range(NUM_SAMPLES):
                angle = np.random.randint(10,70) * (np.pi / 180)
                arrow_len = np.random.randint(15,31)

                if category == "S":
                    bot_len = np.random.randint(120,211)
                    top_len = bot_len - np.random.randint(2,63)
                else:
                    top_len = np.random.randint(120,211)
                    bot_len = top_len - np.random.randint(2,63)

                img = gen_img(top_len, bot_len, arrow_len, angle, type)
                img.save(base_dir+"/"+str(idx)+"_"+category+".jpg")
                idx += 1
    

