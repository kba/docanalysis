import sys
import os
import shutil

import torch
from ocrd import Processor
from ocrd_modelfactory import page_from_file

from ..constants import OCRD_TOOL


# TODO: Change the hardcoded pix2pixHD path (Currently it is a constant due to ocr-d core issue)


class OcrdAnybaseocrDewarper(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-anybaseocr-dewarp']
        kwargs['version'] = OCRD_TOOL['version']
        super(OcrdAnybaseocrDewarper, self).__init__(*args, **kwargs)

    def process(self):
        if torch.cuda.is_available():
            for (n, input_file) in enumerate(self.input_files):
                pcgts = page_from_file(
                    self.workspace.download_file(input_file))
                fname = pcgts.get_Page().imageFilename
                img_tmp_dir = "OCR-D-IMG/test_A"
                #  base = str(fname).split('.')[0]
                img_dir = os.path.dirname(str(fname))
                path = "/home/rakshith/git/pix2pixHD"
                # print(base,img_dir)
                # print(self.parameter['pix2pixHD'])
                os.system("mkdir -p %s" % img_tmp_dir)
                os.system("cp %s %s" % (str(fname), os.path.join(
                    img_tmp_dir, os.path.basename(str(fname)))))
            os.system("python " + path + "/test.py --dataroot %s --checkpoints_dir ./ --name models --results_dir %s --label_nc 0 --no_instance --no_flip --resize_or_crop none --n_blocks_global 10 --n_local_enhancers 2 --gpu_ids %s --loadSize %d --fineSize %d --resize_or_crop %s" %
                      (os.path.dirname(img_tmp_dir), img_dir, self.parameter['gpu_id'], self.parameter['resizeHeight'], self.parameter['resizeWidth'], self.parameter['imgresize']))
            shutil.rmtree(img_tmp_dir)

        else:
            print("Your system has no CUDA installed. No GPU detected.")
            sys.exit(0)
