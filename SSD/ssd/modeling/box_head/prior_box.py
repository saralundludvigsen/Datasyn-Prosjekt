import torch
from math import sqrt
from itertools import product


class PriorBox:
    def __init__(self, cfg):
        self.image_size = cfg.INPUT.IMAGE_SIZE
        prior_config = cfg.MODEL.PRIORS
        self.feature_maps = prior_config.FEATURE_MAPS
        self.min_sizes = prior_config.MIN_SIZES
        self.max_sizes = prior_config.MAX_SIZES
        self.strides = prior_config.STRIDES
        self.aspect_ratios = prior_config.ASPECT_RATIOS
        self.clip = prior_config.CLIP

    def __call__(self):
        """Generate SSD Prior Boxes.
            It returns the center, height and width of the priors. The values are relative to the image size
            Returns:
                priors (num_priors, 4): The prior boxes represented as [[center_x, center_y, w, h]]. All the values
                    are relative to the image size.
        """
        """
        cfg.INPUT.IMAGE_SIZE = [270, 360]
        FEATURE_MAPS: [[34,45], [17, 21], [9, 12], [5, 6], [3, 3], [1, 1]]
        STRIDES: [[8, 8], [16, 27], [30, 30] , [54, 60], [90, 120], [270, 360]
        MIN_SIZES: [[19, 25], [41, 54], [90, 119], [138, 184], [186, 248], [235, 313]]
        MAX_SIZES: [[41, 54], [90, 119], [138, 184], [186, 248], [235, 313], [283, 378]]
        ASPECT_RATIOS: [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        BOXES_PER_LOCATION: [4, 6, 6, 6, 4, 4]
        """
        priors = []
        for k, f in enumerate(self.feature_maps):
            scaley = self.image_size[0] / self.strides[k][0]
            scalex = self.image_size[1] / self.strides[k][1]

            for i, j in product(range(f[0]), range(f[1])):
                
                cx = (j + 0.5) / scalex
                cy = (i + 0.5) / scaley

                # small sized square box
                h_size = self.min_sizes[k][0]
                w_size = self.min_sizes[k][1]
                h = h_size / self.image_size[0]
                w = w_size / self.image_size[1]
                priors.append([cx, cy, w, h])

                # big sized square box
                h_size = sqrt(self.min_sizes[k][0] * self.max_sizes[k][0])
                w_size = sqrt(self.min_sizes[k][1] * self.max_sizes[k][1])
                h = h_size / self.image_size[0]
                w = w_size / self.image_size[1]
                priors.append([cx, cy, w, h])

                # change h/w ratio of the small sized box
                size = self.min_sizes[k]
                h_size = self.min_sizes[k][0]
                w_size = self.min_sizes[k][1]
                h = h_size / self.image_size[0]
                w = w_size / self.image_size[1]
                for ratio in self.aspect_ratios[k]:
                    ratio = sqrt(ratio)
                    priors.append([cx, cy, w * ratio, h / ratio])
                    priors.append([cx, cy, w / ratio, h * ratio])

        priors = torch.tensor(priors)
        if self.clip:
            priors.clamp_(max=1, min=0)
        return priors
