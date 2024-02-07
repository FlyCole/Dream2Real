import os
import pdb
from PIL import Image
import cv2
import numpy as np
import requests
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import warnings
import json
from segmentation.sam_seg import get_thumbnail, mask_touches_edge
from vis_utils import visimg
import gc

os.environ['BITSANDBYTES_NOWELCOME'] = '1'

class Captioner():
    def __init__(self, topdown, device='cuda:0', read_cache=False, cache_path=None):
        self.processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b-coco")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                "Salesforce/blip2-opt-2.7b-coco", load_in_8bit=True, device_map='auto',
            )
        self.batch_size = 200
        self.read_cache = read_cache
        self.cache_path = cache_path
        self.topdown = topdown

    # imgs is a list of PIL images.
    def caption(self, imgs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with torch.no_grad():
                num_batches = int(np.ceil(len(imgs) / self.batch_size))
                captions = []
                print('Captioning objects from all views...')
                with tqdm(total=len(imgs)) as pbar:
                    for batch_idx in range(num_batches):
                        batch_imgs = imgs[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size]
                        inputs = self.processor(images=batch_imgs, return_tensors="pt").to(self.model.device, torch.float16)
                        generated_ids = self.model.generate(**inputs)
                        batch_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                        captions.extend(batch_captions)
                        pbar.update(len(batch_imgs))
                    pbar.close()
                captions = [caption.strip() for caption in captions]
                return captions

    # Return a caption for each obj, aggregated across frames.
    # lang_model is only for aggregation across views.
    # Input RGBs are in HWC format.
    # For scene_masks, 0 means inside scene, 1 means outside scene.
    def caption_objs(self, num_objs, rgbs, masks, lang_model, scene_masks, topdown, multi_view=True, single_view_idx=0):
        if self.read_cache:
            print('Using cached captions')
            agg_captions = json.load(open(self.cache_path, 'r'))
            debug_thumbnails = [None] * len(agg_captions)
            return agg_captions, debug_thumbnails

        print('Creating masked object thumbnails for captioning...')
        all_thumbnails = []
        noise = torch.tensor(np.random.uniform(0, 256, rgbs[0].shape).astype(np.uint8)).to(rgbs.device)
        for obj_idx in range(1, num_objs): # Skip captioning bground.
            obj_thumbnails = []
            frame_range = range(len(rgbs)) if multi_view else [single_view_idx]

            is_container = False
            for frame_idx in frame_range:
                frame_mask = masks[frame_idx].clone()
                rgb_frame = rgbs[frame_idx].clone()
                scene_mask = torch.logical_not(scene_masks[frame_idx].to(frame_mask.device).clone())
                # Since for 3D scenes, first few views will be sideways to fit in more objs in camera view.
                if ((frame_idx in [0, 1]) and not topdown) or (not multi_view and single_view_idx > 0):
                    # Apply rot90 counter-clockwise.
                    rgb_frame = rgb_frame.transpose(0, 1).flip(0)
                    frame_mask = frame_mask.transpose(0, 1).flip(0)
                    scene_mask = scene_mask.transpose(0, 1).flip(0)

                obj_mask = frame_mask == obj_idx
                obj_mask &= scene_mask
                if obj_mask.sum() < 200:
                    continue
                # Avoid captioning objs near img edge, if choice exists.
                if mask_touches_edge(obj_mask) and len(obj_thumbnails) >= 3 and not self.topdown:
                    continue

                # For components of bg mask (not obj mask) which are disconnected from edge,
                # they could be other objects inside the current object.
                # To prevent captioner from detecting them, fill these areas with noise (well, just fill whole bg for efficiency)
                # For efficiency, only do this for objects which are possible containers based on frame 0.
                if frame_idx == 0:
                    num_comps, comp_img = cv2.connectedComponents(np.logical_not(obj_mask.to('cpu').numpy()).astype(np.uint8))
                    for i in range(1, num_comps):
                        comp_mask = comp_img == i
                        black_pix_mask = frame_mask == 0
                        and_part = comp_mask & black_pix_mask.cpu().numpy()
                        intersection = and_part.sum()
                        union = comp_mask.sum()
                        if intersection / union > 0.7:
                            continue
                        if comp_mask.sum() < 400:
                            continue
                        if not mask_touches_edge(torch.tensor(comp_mask)):
                            is_container = True

                if is_container:
                    bg_mask = torch.logical_not(obj_mask)
                    # We want to make the inside of the mask blobby, to destroy the shapes of the objects in the container.
                    # Currently bg_mask will be bg of the container obj, so will have 1s at the inside objects.
                    bg_mask = cv2.GaussianBlur(bg_mask.to('cpu').numpy().astype(np.uint8), (201, 201), 0)
                    bg_mask = cv2.dilate(bg_mask, np.ones((60, 60), np.uint8), iterations=1)
                    bg_mask = torch.tensor(bg_mask).bool().to(obj_mask.device)

                    # Alternative approach:
                    # bg_mask = cv2.erode(bg_mask.to('cpu').numpy().astype(np.uint8), np.ones((40, 40), np.uint8), iterations=1)
                    # bg_mask = torch.tensor(cv2.dilate(bg_mask, np.ones((40, 40), np.uint8), iterations=1))

                    if frame_idx in [0, 1] and not topdown:
                        frame_noise = noise.transpose(0, 1).flip(0)
                    else:
                        frame_noise = noise
                    rgb_frame[bg_mask] = frame_noise[bg_mask]

                mask_for_thumbnail = torch.ones_like(obj_mask) if is_container else obj_mask
                thumbnail = get_thumbnail(rgb_frame, mask_for_thumbnail)
                thumbnail = Image.fromarray(thumbnail.to('cpu').numpy().astype(np.uint8)).convert('RGB')
                obj_thumbnails.append(thumbnail)
            all_thumbnails.append(obj_thumbnails)

        all_captions = []
        batched_thumbnails_imgs = []
        batched_thumbnails_idxs = []
        for obj_idx, obj_thumbnails in enumerate(all_thumbnails):
            batched_thumbnails_imgs.extend(obj_thumbnails)
            batched_thumbnails_idxs.extend([obj_idx] * len(obj_thumbnails))
        batched_captions = self.caption(batched_thumbnails_imgs)

        all_captions = []
        for obj_idx in range(len(all_thumbnails)):
            obj_captions = []
            for caption_idx, caption in enumerate(batched_captions):
                if batched_thumbnails_idxs[caption_idx] == obj_idx:
                    obj_captions.append(caption)
            all_captions.append(obj_captions)

        debug_thumbnails = [obj_thumbnails[0] for obj_thumbnails in all_thumbnails]
        debug_thumbnails.insert(0, rgbs[0])

        if not multi_view:
            print('Using single-view captions')
            agg_captions = [obj_captions[0] for obj_captions in all_captions]
            agg_captions.insert(0, '__background__')
            if self.cache_path is not None:
                json.dump(agg_captions, open(self.cache_path, 'w'))
            return agg_captions, debug_thumbnails

        print('Aggregating captions across views...')
        agg_captions = []
        for obj_captions in tqdm(all_captions):
            agg_caption = lang_model.aggregate_captions_for_obj(obj_captions, silent=True)
            agg_captions.append(agg_caption)
        agg_captions.insert(0, '__background__')

        if self.cache_path is not None:
            json.dump(agg_captions, open(self.cache_path, 'w'))

        return agg_captions, debug_thumbnails

    def free(self):
        # print(f'Memory usage before freeing Captioner: {torch.cuda.memory_allocated(0)}')
        del self.processor
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        # print(f'Memory usage after freeing Captioner: {torch.cuda.memory_allocated(0)}')
