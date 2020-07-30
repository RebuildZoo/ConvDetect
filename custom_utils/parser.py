# from https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/utils/parse_config.py

def parse_model(modeldef_cfg_filename) -> list:
    '''
    Parses the yolo-v3 layer configuration file and returns module definitions
    '''
    module_defs_Lst = []
    with open(modeldef_cfg_filename, "r") as cfg_file:
        for line_i in cfg_file.readlines():
            line_i = line_i.strip()
            if line_i == "" or line_i.startswith("#") : 
                continue 
            if line_i.startswith('['): 
                # This marks the start of a new block
                module_defs_Lst.append({}) # add a new dic; 
                module_defs_Lst[-1]['type'] = line_i[1:-1].rstrip()
                if module_defs_Lst[-1]['type'] == 'convolutional':
                    module_defs_Lst[-1]['batch_normalize'] = 0 # default val
            else:
                key, value = line_i.split("=")
                value = value.strip()
                module_defs_Lst[-1][key.rstrip()] = value.strip()

    return module_defs_Lst


    



if __name__ == "__main__":
    
    cfg_filename = r"D:\Documents\Git_Hub\TneitaP_repo\yoloV3\official_yolo_files\config\yolov3.cfg"

    parse_model(cfg_filename)
