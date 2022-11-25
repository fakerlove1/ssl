import math
import pdb
import numpy as np
import scipy.stats
import torch, torch.nn.functional as F
import albumentations as A
from torchvision.transforms import transforms


class MaskGenerator(object):
    """
    Mask Generator
    """

    def generate_params(self, n_masks, mask_shape, rng=None):
        raise NotImplementedError('Abstract')

    def append_to_batch(self, *batch):
        x = batch[0]
        params = self.generate_params(len(x), x.shape[2:4])
        return batch + (params,)

    def torch_masks_from_params(self, t_params, mask_shape, torch_device):
        raise NotImplementedError('Abstract')


class BoxMaskGenerator(MaskGenerator):
    def __init__(self, prop_range, n_boxes=1, random_aspect_ratio=True, prop_by_area=True, within_bounds=True,
                 invert=False):
        if isinstance(prop_range, float):
            prop_range = (prop_range, prop_range)
        self.prop_range = prop_range
        self.n_boxes = n_boxes
        self.random_aspect_ratio = random_aspect_ratio
        self.prop_by_area = prop_by_area
        self.within_bounds = within_bounds
        self.invert = invert

    def generate_params(self, n_masks, mask_shape, rng=None):
        """
        Box masks can be generated quickly on the CPU so do it there.
        >>> boxmix_gen = BoxMaskGenerator((0.25, 0.25))
        >>> params = boxmix_gen.generate_params(256, (32, 32))
        >>> t_masks = boxmix_gen.torch_masks_from_params(params, (32, 32), 'cuda:0')
        :param n_masks: number of masks to generate (batch size)
        :param mask_shape: Mask shape as a `(height, width)` tuple
        :param rng: [optional] np.random.RandomState instance
        :return: masks: masks as a `(N, 1, H, W)` array
        """
        if rng is None:
            rng = np.random

        if self.prop_by_area:
            # 选择应高于阈值的每个掩码的比例
            # Choose the proportion of each mask that should be above the threshold
            mask_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))

            # Zeros will cause NaNs, so detect and suppres them
            zero_mask = mask_props == 0.0

            if self.random_aspect_ratio:
                y_props = np.exp(rng.uniform(low=0.0, high=1.0, size=(n_masks, self.n_boxes)) * np.log(mask_props))
                x_props = mask_props / y_props
            else:
                y_props = x_props = np.sqrt(mask_props)
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

            y_props[zero_mask] = 0
            x_props[zero_mask] = 0
        else:
            if self.random_aspect_ratio:
                y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
                x_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            else:
                x_props = y_props = rng.uniform(self.prop_range[0], self.prop_range[1], size=(n_masks, self.n_boxes))
            fac = np.sqrt(1.0 / self.n_boxes)
            y_props *= fac
            x_props *= fac

        sizes = np.round(np.stack([y_props, x_props], axis=2) * np.array(mask_shape)[None, None, :])

        # 是否包含变黄
        if self.within_bounds:
            positions = np.round((np.array(mask_shape) - sizes) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(positions, positions + sizes, axis=2)
        else:
            centres = np.round(np.array(mask_shape) * rng.uniform(low=0.0, high=1.0, size=sizes.shape))
            rectangles = np.append(centres - sizes * 0.5, centres + sizes * 0.5, axis=2)

        if self.invert:
            masks = np.zeros((n_masks, 1) + mask_shape)
        else:
            masks = np.ones((n_masks, 1) + mask_shape)
        for i, sample_rectangles in enumerate(rectangles):
            for y0, x0, y1, x1 in sample_rectangles:
                masks[i, 0, int(y0):int(y1), int(x0):int(x1)] = 1 - masks[i, 0, int(y0):int(y1), int(x0):int(x1)]
        return masks

    def torch_masks_from_params(self, t_params, mask_shape, torch_device):
        return t_params


class AddMaskParamsToBatch(object):
    """
    We add the cut-and-paste parameters to the mini-batch within the collate function,
    (we pass it as the `batch_aug_fn` parameter to the `SegCollate` constructor)
    as the collate function pads all samples to a common size
    """

    def __init__(self, mask_gen):
        self.mask_gen = mask_gen

    def __call__(self, batch):
        n_masks = len(batch)
        batch = list(zip(*batch))
        img = torch.FloatTensor(batch[0])
        labels = torch.LongTensor(batch[1])
        mask_size = img.shape[2:]
        params = self.mask_gen.generate_params(n_masks=n_masks, mask_shape=mask_size)
        params = torch.FloatTensor(params)

        # sample = batch[0]
        # # mask_size 等于 [H,W]
        # mask_size = sample['data'].shape[1:3]
        # params = self.mask_gen.generate_params(n_masks=len(batch),mask_shape=mask_size)
        # for sample, p in zip(batch, params):
        #     sample['mask_params'] = p.astype(np.float32)

        return img, labels, params



if __name__ == '__main__':
    x = torch.randn(4, 3, 224, 224)


    class config:
        cutmix_mask_prop_range = (0.25, 0.5)
        cutmix_boxmask_n_boxes = 3
        cutmix_boxmask_fixed_aspect_ratio = False
        cutmix_boxmask_by_size = False
        cutmix_boxmask_outside_bounds = False
        cutmix_boxmask_no_invert = False


    mask_generator = BoxMaskGenerator(prop_range=config.cutmix_mask_prop_range, n_boxes=config.cutmix_boxmask_n_boxes,
                                      random_aspect_ratio=not config.cutmix_boxmask_fixed_aspect_ratio,
                                      prop_by_area=not config.cutmix_boxmask_by_size,
                                      within_bounds=not config.cutmix_boxmask_outside_bounds,
                                      invert=not config.cutmix_boxmask_no_invert)
    add_mask_params_to_batch = AddMaskParamsToBatch(mask_generator)

    from voc import VOC, show, show_label
    from torch.utils.data import DataLoader

    label = 1464
    root = r"E:\note\ssl\data\voc_aug_2\VOCdevkit\VOC2012"
    crop_size = (512, 512)
    val_transform = A.Compose([
        A.PadIfNeeded(min_height=crop_size[0], min_width=crop_size[1], border_mode=0, value=0, p=1, mask_value=255),
        A.RandomCrop(height=crop_size[0], width=crop_size[1]),
        # A.RandomResizedCrop(height=crop_size[0], width=crop_size[1]),  # 随机裁剪缩放
        A.HorizontalFlip(p=0.5),  # 随机水平翻转
        A.ColorJitter(p=1),  # 随机 改变亮度，饱和度
        A.RandomBrightnessContrast(p=0.2),
        # A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ToTensorV2(),
    ])

    test_dataset = VOC(root=root, mode="test", transform=val_transform, label=label)
    unsupervised_train_loader_0 = DataLoader(test_dataset, batch_size=4, shuffle=True, collate_fn=add_mask_params_to_batch)
    unsupervised_train_loader_1 = DataLoader(test_dataset, batch_size=4, shuffle=True)

    unsupervised_dataloader_0 = iter(unsupervised_train_loader_0)
    unsupervised_dataloader_1 = iter(unsupervised_train_loader_1)

    unsup_imgs_0,unsup_label_0,mask_params = unsupervised_dataloader_0.next()
    unsup_imgs_1,unsup_label_1 = unsupervised_dataloader_1.next()
    batch_mix_masks = mask_params
    unsup_imgs_mixed = unsup_imgs_0 * (1 - batch_mix_masks) + unsup_imgs_1 * batch_mix_masks
    show(unsup_imgs_0[0])
    show(unsup_imgs_1[0])
    show(unsup_imgs_mixed[0])







    # for sample in test_dataloader:
    #     print(sample)
    #     break
    # img=sample["data"]
    # label=sample["label"]
    # show(img[0])
    # show(img[1])
    # show(img[2])
    # show_label(label[0].numpy())
    # break
