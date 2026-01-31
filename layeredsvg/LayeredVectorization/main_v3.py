"""
Experimental V3 Layered Vectorization with SDXL + VTracer

This version uses:
- SDXL (1024x1024) instead of SD 1.5 (512x512) for better detail preservation
- VTracer-inspired path simplification techniques
- Aspect ratio preservation via padding

Key changes from v2:
- Uses sds_image_simplicity_sdxl.py for 1024x1024 SDS simplification
- All internal processing at 1024x1024 resolution
- Better quality for high-resolution and non-square images

Usage:
    python main_v3.py -c config/base_config.yaml -timg input.png -fsn output_name
"""

import torch
import torch.nn.functional as F
from PIL import Image
import argparse
from utils.img_process_v3 import *
import os
from tqdm import tqdm
# Use SDXL version for 1024x1024 processing
from sds_image_simplicity_sdxl import sds_based_simplification

# Resolution for v3 (SDXL)
V3_RESOLUTION = 1024
import pydiffvg
import yaml

def init_diffvg(device: torch.device,
                use_gpu: bool = torch.cuda.is_available(),
                print_timing: bool = False):
    pydiffvg.set_device(device)
    pydiffvg.set_use_gpu(use_gpu)
    pydiffvg.set_print_timing(print_timing)

def init_optimizer(shapes,shape_groups,
                   is_train_stroke: bool = False,
                   is_train_color: bool = True,
                   is_opt_list: List[int] = [],
                   lr_base: dict = {}):
    params = {}
    points_vars = []
    color_vars = []

    stroke_width_vars = []
    stroke_color_vars = []

    if len(is_opt_list) == 0:
        is_opt_list = [1 for i in range(len(shapes))]

    for i, path in enumerate(shapes):
        if is_opt_list[i]==1:
            path.id = i  # set point id
            path.points.requires_grad = True
            points_vars.append(path.points)
            if is_train_stroke:
                path.stroke_width.requires_grad = True
                stroke_width_vars.append(path.stroke_width)
    if is_train_color:
        for i, group in enumerate(shape_groups):
            if is_opt_list[i]==1:
                group.fill_color.requires_grad = True
                color_vars.append(group.fill_color)
                if is_train_stroke:
                    group.stroke_color.requires_grad = True
                    stroke_color_vars.append(group.stroke_color)

    params = {}
    params['point'] = points_vars
    if is_train_color:
        params['color'] = color_vars
    if is_train_stroke:
        params['stroke_width'] = stroke_width_vars
        params['stroke_color'] = stroke_color_vars

    learnable_params = [
        {'params': params[ki], 'lr': lr_base[ki], '_id': str(ki)} for ki in sorted(params.keys())
    ]
    svg_optimizer = torch.optim.Adam(learnable_params, betas=(0.9, 0.9), eps=1e-6)
    return svg_optimizer

def exclude_loss(raster_img,scale=1):
    img = F.relu(178/255 - raster_img)
    # raster_img_1 = transforms.ToPILImage()(raster_img)
    # raster_img_1.save(f"{time.time()}.png")
    loss = torch.sum(img)*scale
    return loss

def svg_optimize_img_struct(device, shapes, shape_groups,
                            target_img: np.ndarray,
                            layerd_struct_masks: list,
                            file_save_path: str,
                            train_conf: dict,
                            base_lr_conf: dict):
    struct_target_imgs, struct_colors_list = init_struct_target_imgs(layerd_struct_masks)
    struct_target_imgs = [x.to(device) for x in struct_target_imgs]
    struct_shape_groups_list = []
    for struct_colors in struct_colors_list:
        struct_shape_groups = []
        for i,color in enumerate(struct_colors):
            path_group = pydiffvg.ShapeGroup(
                            shape_ids=torch.LongTensor([i]),
                            fill_color=torch.FloatTensor(color+[1]),
                            stroke_color=torch.FloatTensor([0,0,0,1])
                            )
            struct_shape_groups.append(path_group)
        struct_shape_groups_list.append(struct_shape_groups)

    transparent_shape_groups = []
    for i in range(len(shapes)):
        path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.LongTensor([i]),
                        fill_color=torch.FloatTensor([0,0,0,0.3]),
                        stroke_color=torch.FloatTensor([0,0,0,0.3])
                    )
        transparent_shape_groups.append(path_group)

    black_bg = torch.tensor([0., 0., 0.], requires_grad=False, device=device)
    white_bg = torch.tensor([1., 1., 1.], requires_grad=False, device=device)

    img_height, img_width = target_img.shape[:2]
    target_img = torch.tensor(target_img,device=device)/255
    target_img = target_img.permute(2, 0, 1)

    svg_optimizer = init_optimizer(shapes,shape_groups,
                                   train_conf["is_train_stroke"],
                                   train_conf["is_train_struct_color"],
                                   lr_base=base_lr_conf)

    with tqdm(total=train_conf["struct_opt_num_iters"], desc="Processing value", unit="value") as pbar:
        for i in range(train_conf["struct_opt_num_iters"]):
            loss_struct = 0
            loss_exclude=0
            shape_index = 0
            for struct_i,struct_target_img in enumerate(struct_target_imgs):
                shape_index+=len(layerd_struct_masks[struct_i])
                struct_img = svg_to_img(img_width,img_height,
                                        shapes[shape_index-len(layerd_struct_masks[struct_i]):shape_index],
                                        struct_shape_groups_list[struct_i],
                                        device)
                struct_img = rgba_to_rgb(struct_img,device,black_bg)
                loss_struct+=F.mse_loss(struct_img, struct_target_img)

                transparent_img = svg_to_img(img_width,img_height,
                                             shapes[shape_index-len(layerd_struct_masks[struct_i]):shape_index],
                                             transparent_shape_groups[:len(layerd_struct_masks[struct_i])],
                                             device)
                transparent_img = rgba_to_rgb(transparent_img,device,white_bg)
                loss_exclude += exclude_loss(transparent_img,scale=2e-7)

            img = svg_to_img(img_width,img_height,shapes,shape_groups,device)
            img = rgba_to_rgb(img,device,white_bg)
            loss_mse = F.mse_loss(img, target_img)

            loss = loss_mse*0.02+loss_exclude+loss_struct
            svg_optimizer.zero_grad()
            loss.backward()
            svg_optimizer.step()
            pydiffvg.save_svg(f"{file_save_path}/{i}.svg",
                            img_width,
                            img_height,
                            shapes,
                            shape_groups)
            pbar.update(1)
    return shapes,shape_groups

def svg_optimize_img_visual(device, shapes, shape_groups,
                            target_img: np.ndarray,
                            file_save_path: str,
                            is_opt_list: List[int],
                            train_conf: dict,
                            base_lr_conf: dict,
                            count: int = 0,
                            struct_path_num: int = 0,
                            is_path_merging_phase: bool = False):

    img_height, img_width = target_img.shape[:2]
    target_img = torch.tensor(target_img,device=device)/255
    target_img = target_img.permute(2, 0, 1)

    transparent_shape_groups = []
    for i in range(len(shapes)-struct_path_num):
        path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.LongTensor([i]),
                        fill_color=torch.FloatTensor([0,0,0,0.3]),
                        stroke_color=torch.FloatTensor([0,0,0,0.3])
                    )
        transparent_shape_groups.append(path_group)

    svg_optimizer = init_optimizer(shapes,shape_groups,
                                   train_conf["is_train_stroke"],
                                   train_conf["is_train_visual_color"],
                                   is_opt_list,
                                   lr_base=base_lr_conf)
    num_iters = train_conf["visual_opt_num_iters"]
    if is_path_merging_phase:
        num_iters = 50

    with tqdm(total=num_iters, desc="Processing value", unit="value") as pbar:
        for i in range(num_iters):
            img = svg_to_img(img_width,img_height,shapes,shape_groups,device)
            img = rgba_to_rgb(img,device)
            loss = F.mse_loss(img, target_img)
            svg_optimizer.zero_grad()
            loss.backward()
            svg_optimizer.step()
            pydiffvg.save_svg(f"{file_save_path}/{count}.svg",
                            img_width,
                            img_height,
                            shapes,
                            shape_groups)
            count+=1
            pbar.update(1)
    return shapes,shape_groups,count

def layered_vectorization(args, device=None):
    """
    Main vectorization pipeline with VTracer enhancements.
    """
    # Get VTracer parameters from args (with defaults)
    vtracer_enable = getattr(args, 'vtracer_enable', True)
    staircase_area = getattr(args, 'staircase_area', 1.5)
    corner_angle = getattr(args, 'corner_angle', 135.0)
    simplify_error = getattr(args, 'simplify_error', 2.0)
    smooth_iterations = getattr(args, 'smooth_iterations', 0)

    if vtracer_enable:
        print(f"VTracer Enhancement: ENABLED")
        print(f"  - Staircase area threshold: {staircase_area}")
        print(f"  - Corner angle threshold: {corner_angle}°")
        print(f"  - Simplification error: {simplify_error}")
        print(f"  - Smooth iterations: {smooth_iterations}")
    else:
        print("VTracer Enhancement: DISABLED (using original approxPolyDP)")

    simp_img_seq_save_path = f"./workdir/{args.file_save_name}/simplified_image_sequence"
    os.makedirs(simp_img_seq_save_path,exist_ok=True)
    all_simp_img_seq_save_path = "-1"
    if args.is_save_all_simp_img_seq:
        all_simp_img_seq_save_path = f"./workdir/{args.file_save_name}/all_simplified_image_sequence"
        os.makedirs(all_simp_img_seq_save_path,exist_ok=True)
    masks_save_path=-1
    if args.is_save_masks:
        masks_save_path = f"./workdir/{args.file_save_name}/masks"
        os.makedirs(masks_save_path,exist_ok=True)
    struct_svgs_save_path = f"./workdir/{args.file_save_name}/struct_svgs"
    os.makedirs(struct_svgs_save_path,exist_ok=True)
    visual_svgs_save_path = f"./workdir/{args.file_save_name}/struct&visual_svgs"
    os.makedirs(visual_svgs_save_path,exist_ok=True)
    layerd_struct_save_path = f"./workdir/{args.file_save_name}/layerd_struct"
    os.makedirs(layerd_struct_save_path,exist_ok=True)

    print("SDS-based Simplification...")
    simp_img_seq = sds_based_simplification(device,
                                            args.target_image,
                                            args.simp_img_seq_indexs,
                                            simp_img_seq_save_path,
                                            all_simp_img_seq_save_path)
    target_img = simp_img_seq[0]
    img_height, img_width = target_img.shape[:2]

    print("SAM...")
    masks = sam_img_seq(device,simp_img_seq,masks_save_path,args.sam)

    print("Layered Structure Reconstruction...")
    # Structural layer optimization.
    layerd_struct_masks = layer_segmented_masks([[masks[0]]],masks[1:])
    layerd_struct_masks = get_struct_masks_by_area(layerd_struct_masks,int(args.max_path_num_limit*0.4))

    # Use VTracer-enhanced path initialization
    shapes, shape_groups = init_svg_by_mask(
        layerd_struct_masks, target_img, args.approxpolydp_epsilon,
        use_vtracer_enhancement=vtracer_enable,
        staircase_area=staircase_area,
        corner_angle=corner_angle,
        simplify_error=simplify_error,
        smooth_iterations=smooth_iterations
    )

    shapes,shape_groups = svg_optimize_img_struct(device,shapes,shape_groups,
                                                  target_img,
                                                  layerd_struct_masks,
                                                  struct_svgs_save_path,
                                                  args.train,
                                                  args.base_lr)
    if args.color_fitting_type not in ["dominan","mse"]:
        raise ValueError(f"args.color_fitting_type can only be 'dominan' or 'mse', but the values passed in are {args.color_fitting_type}")
    if args.color_fitting_type == "dominan":
        shape_groups,target_img_cluster = color_fitting(shape_groups, target_img, layerd_struct_masks, args.is_cluster_target_img, args.kmeas_k)
        target_img_cluster_ = Image.fromarray(target_img_cluster)
        target_img_cluster_.save(f"./workdir/{args.file_save_name}/cluster_img.png")
        pydiffvg.save_svg(f"./workdir/{args.file_save_name}/color-adjusted.svg",img_height,img_width,shapes,shape_groups)

    print("Visual Refinement...")
    pseudo_struct_masks = [mask for sublist in layerd_struct_masks for mask in sublist]
    is_opt_list = []
    count = 0
    struct_path_num = len(shapes)
    for i in range(args.add_visual_path_num_iters):
        os.makedirs(f"{visual_svgs_save_path}/{i}_add_paths",exist_ok=True)
        if i == args.add_visual_path_num_iters-1:
            remaining_path_num = args.max_path_num_limit-len(shapes)
        else:
            remaining_path_num = int((args.max_path_num_limit-len(shapes))*0.6)

        # Use VTracer-enhanced visual path addition
        shapes, shape_groups, pseudo_struct_masks, is_opt_list, struct_path_num = add_visual_paths(
            shapes, shape_groups, device,
            struct_path_num,
            target_img_cluster,
            pseudo_struct_masks,
            is_opt_list,
            epsilon=args.approxpolydp_epsilon,
            N=remaining_path_num,
            use_vtracer_enhancement=vtracer_enable,
            staircase_area=staircase_area,
            corner_angle=corner_angle,
            simplify_error=simplify_error,
            smooth_iterations=smooth_iterations
        )

        if struct_path_num == -1:
            print("There are no new paths to add.")
            break

        print("Add new path")
        shapes,shape_groups,count = svg_optimize_img_visual(device,shapes,shape_groups,
                                                            target_img,
                                                            f"{visual_svgs_save_path}/{i}_add_paths",
                                                            is_opt_list,
                                                            args.train,
                                                            args.base_lr,
                                                            count,
                                                            struct_path_num)
        if i == args.add_visual_path_num_iters-1:
            break
        shapes,shape_groups = remove_lowquality_paths(shapes, shape_groups, device,
                                                      img_width, img_height,
                                                      visual_difference_threshold=args.paths_remove_visual_threshold,
                                                      struct_path_num=struct_path_num)

        print("Path merging")
        os.makedirs(f"{visual_svgs_save_path}/{i}_merge_paths",exist_ok=True)
        shapes,shape_groups,pseudo_struct_masks,is_opt_list,struct_path_num = merge_path(shapes, shape_groups,device,
                                                                                            img_width, img_height,
                                                                                            struct_path_num,
                                                                                            pseudo_struct_masks,
                                                                                            is_opt_list,
                                                                                            color_threshold=args.paths_merge_color_threshold,
                                                                                            overlapping_area_threshold=args.paths_merge_distance_threshold)
        shapes,shape_groups,count = svg_optimize_img_visual(device,shapes,shape_groups,
                                                    target_img,
                                                    f"{visual_svgs_save_path}/{i}_merge_paths",
                                                    is_opt_list,
                                                    args.train,
                                                    args.base_lr,
                                                    count,
                                                    struct_path_num,
                                                    is_path_merging_phase=True)
    pydiffvg.save_svg(f"./workdir/{args.file_save_name}/final.svg",img_height,img_width,shapes,shape_groups)

def load_config(file_path,args):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
        for key, value in config.items():
            setattr(args, key, value)
    return args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Layered Image Vectorization with VTracer Enhancement")
    parser.add_argument("-c", "--config", type=str, default="./config/base_config.yaml",
                        help="YAML/YML file for configuration.")
    parser.add_argument("-timg", "--target_image", type=str, required=True,
                        help="Path to target image.")
    parser.add_argument("-fsn", "--file_save_name", type=str, default="output",
                        help="Files save name.")

    # VTracer enhancement parameters
    parser.add_argument("--vtracer_enable", type=bool, default=True,
                        help="Enable VTracer enhancements (default: True)")
    parser.add_argument("--staircase_area", type=float, default=1.5,
                        help="Threshold for staircase removal (default: 1.5)")
    parser.add_argument("--corner_angle", type=float, default=135.0,
                        help="Angle threshold for corner detection in degrees (default: 135.0)")
    parser.add_argument("--simplify_error", type=float, default=2.0,
                        help="Max error for path simplification (default: 2.0)")
    parser.add_argument("--smooth_iterations", type=int, default=0,
                        help="Number of 4-point subdivision smoothing passes (default: 0)")

    args = parser.parse_args()
    args = load_config(args.config, args)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init_diffvg(device=device)
    layered_vectorization(args, device)
